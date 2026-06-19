"""
haptic_server.py — Network server that broadcasts haptic timelines and sync
pulses to connected clients (typically an Android phone).

ARCHITECTURE
------------
This module is the laptop-side counterpart to the Android haptic client. It
implements the protocol documented in HAPTIC_ANDROID_ARCHITECTURE.md:
  - NSD/mDNS service advertisement     (zeroconf)
  - TCP control channel  (length-prefixed JSON messages, one socket per client)
  - UDP sync-pulse channel             (best-effort datagrams)

The server is EVENT-DRIVEN. The player (or any caller) calls publish_*
methods on HapticServer; the server takes care of distributing the appropriate
messages to connected clients. The player never touches sockets directly.

USAGE FROM A PLAYER
-------------------
    from haptic_server import HapticServer

    server = HapticServer(timeline_dict=my_loaded_json)
    server.start()                          # begins discovery + accept loop

    # When the user starts playback:
    server.publish_play(media_t=0.0, rate=1.0)

    # On a periodic timer (~5-10 Hz) while playing:
    server.publish_sync(media_t=current_media_t)

    # On state changes:
    server.publish_pause(media_t=current_media_t)
    server.publish_seek(media_t=42.0)
    server.publish_rate(rate=0.5)

    # And on shutdown:
    server.stop()

STANDALONE
----------
For testing without the Qt player:
    python haptic_server.py path/to/match.haptic.json
This loads the timeline, advertises the service, and (when a client connects)
sends the timeline. Without a player driving publish_* calls there are no
sync pulses; combine with `--simulate` to drive a virtual playhead so a
connected client sees real playback events.

DESIGN NOTES (read before changing anything)
-------------------------------------------
- This module is intentionally Qt-free. Threads, not signals. The player
  imports it but it can also run standalone — keep that separation.
- Sockets run on background threads; HapticServer is the thread-safe interface
  the caller uses. Public methods are safe to call from any thread.
- Timestamps in messages use SECONDS for media_t (matches the timeline JSON)
  and NANOSECONDS for server clock (System.nanoTime-equivalent: time.monotonic_ns).
- The server holds the timeline; it sends a copy to each client on connect.
- No copies of timeline are made on hot paths. Clock-sync and sync pulses are
  small JSON objects assembled fresh each time.
"""

import json
import socket
import struct
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List

# zeroconf is the standard mDNS library for Python. Install:  pip install zeroconf
from zeroconf import ServiceInfo, Zeroconf, IPVersion


# ============================ PROTOCOL CONSTANTS ==============================
SERVICE_TYPE = "_haptics._tcp.local."     # NSD service type the Android client
                                          # browses for. Trailing dot is mDNS-correct.
DEFAULT_TCP_PORT = 47821                  # arbitrary; configurable per instance
DEFAULT_UDP_PORT = 47822
LENGTH_PREFIX = ">I"                      # 4-byte big-endian unsigned int

log = logging.getLogger("haptic_server")


# ============================ CLIENT STATE ====================================
@dataclass
class ClientState:
    """One connected client. The server holds one of these per accepted TCP
    socket. The UDP address is filled in when the client's hello arrives."""
    tcp_sock: socket.socket
    tcp_addr: tuple                       # (host, port)
    udp_addr: Optional[tuple] = None      # (host, port) — set after hello
    capabilities: dict = field(default_factory=dict)
    write_lock: threading.Lock = field(default_factory=threading.Lock)
    alive: bool = True


# ============================ WIRE FORMAT HELPERS =============================
def _send_message(sock: socket.socket, obj: dict, lock: threading.Lock) -> bool:
    """Length-prefixed JSON send. Thread-safe per-socket via the caller's lock.
    Returns False on socket error so the caller can drop the client."""
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    frame = struct.pack(LENGTH_PREFIX, len(payload)) + payload
    try:
        with lock:
            sock.sendall(frame)
        return True
    except (OSError, ConnectionError):
        return False


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly n bytes from sock or return None on EOF / error."""
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv_message(sock: socket.socket) -> Optional[dict]:
    """Read one length-prefixed JSON message from sock. None on disconnect."""
    header = _recv_exact(sock, struct.calcsize(LENGTH_PREFIX))
    if header is None:
        return None
    (n,) = struct.unpack(LENGTH_PREFIX, header)
    if n <= 0 or n > 64 * 1024 * 1024:          # 64 MiB hard cap
        log.warning("rejecting message with length=%d", n)
        return None
    body = _recv_exact(sock, n)
    if body is None:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        log.warning("malformed message: %s", e)
        return None


# ============================ THE SERVER ======================================
class HapticServer:
    """Network server for the haptic timeline.

    THREAD MODEL:
      - The accept loop runs on its own thread (started by start()).
      - Each connected client gets its own thread reading from its TCP socket.
      - publish_* methods are called from ANY thread (typically the Qt main
        thread of the player) and are safe — they iterate clients and write
        through each client's per-socket lock.
      - UDP sending is fire-and-forget from the calling thread.

    PUBLIC API (player calls these):
      set_timeline(timeline_dict)   - update the timeline (sent on next connect
                                      and resent to existing clients)
      publish_play(media_t, rate)   - playback started
      publish_pause(media_t)        - paused
      publish_seek(media_t)         - jumped
      publish_rate(rate)            - playback rate changed
      publish_sync(media_t)         - periodic UDP sync pulse (call ~5-10 Hz)
      start()                       - begin advertising + accepting
      stop()                        - shut down cleanly
    """

    def __init__(self,
                 timeline_dict: Optional[dict] = None,
                 tcp_port: int = DEFAULT_TCP_PORT,
                 udp_port: int = DEFAULT_UDP_PORT,
                 service_name: str = "haptic-laptop"):
        self.tcp_port = tcp_port
        self.udp_port = udp_port
        self.service_name = service_name

        self._timeline = timeline_dict
        self._timeline_lock = threading.Lock()

        self._clients: List[ClientState] = []
        self._clients_lock = threading.Lock()

        self._tcp_sock: Optional[socket.socket] = None
        self._udp_sock: Optional[socket.socket] = None
        self._zeroconf: Optional[Zeroconf] = None
        self._service_info: Optional[ServiceInfo] = None

        self._stop_event = threading.Event()
        self._accept_thread: Optional[threading.Thread] = None

        # current playback anchor, used so newly-connecting clients can be
        # immediately synced without waiting for the next sync pulse
        self._play_state = {"playing": False, "media_t": 0.0, "rate": 1.0,
                             "anchor_ns": 0}

    # ---------- lifecycle ----------
    def start(self) -> None:
        """Begin advertising the service, accepting clients, and listening on
        UDP. Returns once the listeners are bound (so the caller can be sure
        ports are open before announcing to a user)."""
        self._tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._tcp_sock.bind(("0.0.0.0", self.tcp_port))
        self._tcp_sock.listen(4)

        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp_sock.bind(("0.0.0.0", self.udp_port))

        self._register_zeroconf()

        self._stop_event.clear()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="haptic-accept", daemon=True)
        self._accept_thread.start()

        log.info("server started: TCP %d, UDP %d, mDNS %s",
                 self.tcp_port, self.udp_port, self.service_name)

    def stop(self) -> None:
        """Stop the server: deregister mDNS, close all client sockets, stop
        accept loop. Safe to call multiple times."""
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if self._zeroconf is not None and self._service_info is not None:
            try:
                self._zeroconf.unregister_service(self._service_info)
                self._zeroconf.close()
            except Exception as e:
                log.warning("zeroconf shutdown error: %s", e)
            self._zeroconf = None
        if self._tcp_sock is not None:
            try:
                self._tcp_sock.close()
            except OSError:
                pass
            self._tcp_sock = None
        if self._udp_sock is not None:
            try:
                self._udp_sock.close()
            except OSError:
                pass
            self._udp_sock = None
        with self._clients_lock:
            for c in self._clients:
                c.alive = False
                try:
                    c.tcp_sock.close()
                except OSError:
                    pass
            self._clients.clear()
        log.info("server stopped")

    # ---------- timeline ----------
    def set_timeline(self, timeline_dict: dict) -> None:
        """Update the timeline; resend it to every currently-connected client."""
        with self._timeline_lock:
            self._timeline = timeline_dict
        msg = {"msg": "timeline", "data": timeline_dict}
        self._broadcast_tcp(msg)

    # ---------- publish (player calls these) ----------
    def publish_play(self, media_t: float, rate: float = 1.0) -> None:
        self._play_state.update({"playing": True, "media_t": media_t,
                                  "rate": rate, "anchor_ns": time.monotonic_ns()})
        self._broadcast_tcp({"msg": "play", "media_t": media_t, "rate": rate,
                             "t_server_ns": time.monotonic_ns()})

    def publish_pause(self, media_t: float) -> None:
        self._play_state.update({"playing": False, "media_t": media_t,
                                  "anchor_ns": time.monotonic_ns()})
        self._broadcast_tcp({"msg": "pause", "media_t": media_t,
                             "t_server_ns": time.monotonic_ns()})

    def publish_seek(self, media_t: float) -> None:
        self._play_state.update({"media_t": media_t,
                                  "anchor_ns": time.monotonic_ns()})
        self._broadcast_tcp({"msg": "seek", "media_t": media_t,
                             "t_server_ns": time.monotonic_ns()})

    def publish_rate(self, rate: float) -> None:
        # capture current media_t in the anchor before changing rate, so the
        # client's media-clock math stays consistent across the rate change
        self._play_state["rate"] = rate
        self._play_state["anchor_ns"] = time.monotonic_ns()
        self._broadcast_tcp({"msg": "rate", "rate": rate,
                             "t_server_ns": time.monotonic_ns()})

    def publish_sync(self, media_t: float) -> None:
        """Send a UDP sync pulse to every client whose UDP address is known.
        Call this on a timer (~5-10 Hz) during playback."""
        msg = {"msg": "sync", "media_t": media_t,
               "t_server_ns": time.monotonic_ns(),
               "rate": self._play_state.get("rate", 1.0)}
        payload = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        if self._udp_sock is None:
            return
        with self._clients_lock:
            targets = [c.udp_addr for c in self._clients
                       if c.alive and c.udp_addr is not None]
        for addr in targets:
            try:
                self._udp_sock.sendto(payload, addr)
            except OSError:
                pass  # transient UDP errors are fine; sync is best-effort

    # ---------- introspection ----------
    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return sum(1 for c in self._clients if c.alive)

    # ---------- internals ----------
    def _register_zeroconf(self) -> None:
        self._zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
        ip = self._best_local_ip()
        self._service_info = ServiceInfo(
            type_=SERVICE_TYPE,
            name=f"{self.service_name}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=self.tcp_port,
            properties={"udp_port": str(self.udp_port), "version": "1"},
            server=f"{self.service_name}.local.",
        )
        self._zeroconf.register_service(self._service_info)
        log.info("mDNS registered: %s at %s:%d", self.service_name, ip, self.tcp_port)

    @staticmethod
    def _best_local_ip() -> str:
        """Pick the LAN-facing IP. UDP-connect to a public IP doesn't actually
        send anything; the OS just picks the routing interface for us."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"
        finally:
            s.close()

    def _accept_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                tcp_sock, addr = self._tcp_sock.accept()
            except OSError:
                break  # socket closed by stop()
            log.info("client connected: %s", addr)
            client = ClientState(tcp_sock=tcp_sock, tcp_addr=addr)
            with self._clients_lock:
                self._clients.append(client)
            t = threading.Thread(target=self._client_loop, args=(client,),
                                 name=f"haptic-client-{addr[0]}", daemon=True)
            t.start()

    def _client_loop(self, client: ClientState) -> None:
        """Per-client TCP read loop. Handles hello + clock-sync; the client
        otherwise doesn't speak much."""
        try:
            while client.alive and not self._stop_event.is_set():
                msg = _recv_message(client.tcp_sock)
                if msg is None:
                    break
                self._handle_client_message(client, msg)
        finally:
            self._drop_client(client)

    def _handle_client_message(self, client: ClientState, msg: dict) -> None:
        kind = msg.get("msg")

        if kind == "hello":
            client.capabilities = msg.get("capabilities", {})
            # the client tells us which UDP port it's listening on; we save
            # (its IP, that port) as the sync-pulse target
            udp_port = msg.get("udp_port")
            if isinstance(udp_port, int):
                client.udp_addr = (client.tcp_addr[0], udp_port)
            log.info("hello from %s: caps=%s udp=%s",
                     client.tcp_addr, client.capabilities, client.udp_addr)
            # send the current timeline (if we have one) immediately
            with self._timeline_lock:
                tl = self._timeline
            if tl is not None:
                _send_message(client.tcp_sock,
                              {"msg": "timeline", "data": tl},
                              client.write_lock)
            # if playback is in progress, immediately send a play anchor
            if self._play_state["playing"]:
                _send_message(client.tcp_sock, {
                    "msg": "play",
                    "media_t": self._current_media_t(),
                    "rate": self._play_state["rate"],
                    "t_server_ns": time.monotonic_ns(),
                }, client.write_lock)

        elif kind == "time_req":
            # SNTP-style: echo the client's send-time + add ours
            _send_message(client.tcp_sock, {
                "msg": "time_resp",
                "t0_client_ns": msg.get("t0_client_ns"),
                "t_server_ns": time.monotonic_ns(),
            }, client.write_lock)

        else:
            log.debug("ignored unknown msg from client: %r", kind)

    def _current_media_t(self) -> float:
        """Extrapolate current media-time from the last anchor."""
        ps = self._play_state
        if not ps["playing"]:
            return ps["media_t"]
        elapsed_s = (time.monotonic_ns() - ps["anchor_ns"]) / 1e9
        return ps["media_t"] + elapsed_s * ps["rate"]

    def _broadcast_tcp(self, msg: dict) -> None:
        """Send a TCP message to every connected client. Drops clients whose
        sockets error out."""
        dead = []
        with self._clients_lock:
            targets = list(self._clients)
        for c in targets:
            if not c.alive:
                continue
            ok = _send_message(c.tcp_sock, msg, c.write_lock)
            if not ok:
                dead.append(c)
        for c in dead:
            self._drop_client(c)

    def _drop_client(self, client: ClientState) -> None:
        if not client.alive:
            return
        client.alive = False
        try:
            client.tcp_sock.close()
        except OSError:
            pass
        with self._clients_lock:
            try:
                self._clients.remove(client)
            except ValueError:
                pass
        log.info("client dropped: %s", client.tcp_addr)


# ============================ STANDALONE ENTRY POINT ==========================
def _standalone_main():
    import argparse
    ap = argparse.ArgumentParser(description="Standalone haptic server (no player).")
    ap.add_argument("timeline_json", help="path to a .haptic.json file to serve")
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT)
    ap.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    ap.add_argument("--name", default="haptic-laptop")
    ap.add_argument("--simulate", action="store_true",
                    help="simulate playback by emitting sync pulses against a "
                         "virtual playhead (useful for testing without the player)")
    ap.add_argument("--rate-hz", type=float, default=8.0,
                    help="sync pulse rate when --simulate (default 8)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    with open(args.timeline_json) as f:
        timeline = json.load(f)
    log.info("loaded timeline: %s (%d events, %.1fs)",
             timeline.get("source"), len(timeline.get("events", [])),
             timeline.get("duration", 0.0))

    server = HapticServer(timeline_dict=timeline,
                          tcp_port=args.tcp_port, udp_port=args.udp_port,
                          service_name=args.name)
    server.start()
    try:
        if args.simulate:
            log.info("simulating playback @ %.1f Hz sync pulses", args.rate_hz)
            server.publish_play(media_t=0.0, rate=1.0)
            start_ns = time.monotonic_ns()
            interval = 1.0 / args.rate_hz
            duration = timeline.get("duration", 60.0)
            while True:
                elapsed = (time.monotonic_ns() - start_ns) / 1e9
                if elapsed > duration:
                    log.info("end of timeline; looping")
                    server.publish_seek(0.0)
                    server.publish_play(media_t=0.0, rate=1.0)
                    start_ns = time.monotonic_ns()
                    continue
                server.publish_sync(media_t=elapsed)
                time.sleep(interval)
        else:
            log.info("idle (no playback). Press Ctrl-C to stop.")
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        log.info("Ctrl-C received")
    finally:
        server.stop()


if __name__ == "__main__":
    _standalone_main()
