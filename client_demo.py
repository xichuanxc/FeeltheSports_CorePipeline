"""
haptic_client_demo.py — A fake Android client that exercises the haptic
server, for testing the network layer without a real phone.

What it does:
  1. Discovers the laptop server via mDNS (looks for _haptics._tcp.local.)
  2. Opens TCP, sends `hello` with fake capabilities
  3. Receives the timeline
  4. Runs the SNTP-style clock-sync handshake
  5. Opens a UDP socket on a random port, listens for sync pulses
  6. Schedules and "plays" haptic events as their media-times arrive
     (printing a clearly-formatted line per event — a real Android client
     would call Vibrator.vibrate() here instead)

It also runs a stdin reader: while connected, pressing `f<ENTER>` triggers a
manual haptic event using the most recent event from the timeline. Pressing
`q<ENTER>` quits.

Usage:
    python haptic_server.py path/to/match.haptic.json --simulate
    # (in another terminal)
    python haptic_client_demo.py

Add --server HOST:PORT to skip NSD discovery and connect directly.
"""

import argparse
import bisect
import json
import logging
import queue
import socket
import struct
import sys
import threading
import time
from typing import Optional, List

from zeroconf import Zeroconf, ServiceBrowser, ServiceListener


SERVICE_TYPE = "_haptics._tcp.local."
LENGTH_PREFIX = ">I"
log = logging.getLogger("haptic_client_demo")


# ============================ WIRE FORMAT =====================================
def send_message(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack(LENGTH_PREFIX, len(payload)) + payload)


def recv_message(sock: socket.socket) -> Optional[dict]:
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            return None
        header += chunk
    (n,) = struct.unpack(LENGTH_PREFIX, header)
    body = b""
    while len(body) < n:
        chunk = sock.recv(n - len(body))
        if not chunk:
            return None
        body += chunk
    return json.loads(body.decode("utf-8"))


# ============================ NSD DISCOVERY ===================================
class _Listener(ServiceListener):
    def __init__(self):
        self.found: Optional[tuple] = None      # (host, port)
        self.event = threading.Event()

    def add_service(self, zc, type_, name):
        info = zc.get_service_info(type_, name)
        if not info:
            return
        host = socket.inet_ntoa(info.addresses[0])
        self.found = (host, info.port)
        log.info("mDNS found: %s at %s:%d", name, host, info.port)
        self.event.set()

    def update_service(self, zc, type_, name): pass
    def remove_service(self, zc, type_, name): pass


def discover(timeout: float = 5.0) -> Optional[tuple]:
    """Returns (host, port) of the haptic server, or None on timeout."""
    zc = Zeroconf()
    listener = _Listener()
    browser = ServiceBrowser(zc, SERVICE_TYPE, listener)
    try:
        if not listener.event.wait(timeout):
            return None
        return listener.found
    finally:
        zc.close()


# ============================ HAPTIC "PLAYBACK" ===============================
# On a real Android client this is where Vibrator.vibrate() lives. Here we
# just print a clearly-formatted line so you can SEE the timing. The intent
# is for these prints to look as similar to what the phone would do as possible.

def fire_haptic(event: dict, now_local_ns: int, target_local_ns: int,
                source: str = "scheduled") -> None:
    """Render one haptic event. `source` distinguishes timeline-scheduled
    fires from manual `f`-trigger fires for easy visual identification."""
    delta_ms = (now_local_ns - target_local_ns) / 1e6
    type_ = event.get("type", "?")
    intensity = float(event.get("intensity", 0.0))
    # Visual "intensity bar"
    bar = "#" * max(1, int(intensity * 20))
    tag = "[MANUAL]   " if source == "manual" else "[scheduled]"
    print(f"  HAPTIC {tag} t={event['time']:7.3f}s  "
          f"type={type_:7s}  intensity={intensity:.2f} {bar:<20}  "
          f"timing-error={delta_ms:+5.1f}ms")


# ============================ MEDIA CLOCK =====================================
class MediaClock:
    """Mirrors the model in the Android architecture doc: anchored to a
    server-clock timestamp and a media-time, advances at `rate`."""
    def __init__(self):
        self.clock_offset_ns = 0          # server_ns - local_ns
        self.anchor_media_t = 0.0
        self.anchor_server_ns = 0
        self.rate = 1.0
        self.playing = False
        self._lock = threading.Lock()

    def set_offset(self, offset_ns: int):
        with self._lock:
            self.clock_offset_ns = offset_ns

    def anchor(self, media_t: float, server_ns: int, rate: float, playing: bool):
        with self._lock:
            self.anchor_media_t = media_t
            self.anchor_server_ns = server_ns
            self.rate = rate
            self.playing = playing

    def media_t_now(self) -> float:
        with self._lock:
            if not self.playing:
                return self.anchor_media_t
            now_server_ns = time.monotonic_ns() + self.clock_offset_ns
            elapsed_s = (now_server_ns - self.anchor_server_ns) / 1e9
            return self.anchor_media_t + elapsed_s * self.rate

    def local_ns_for_media_t(self, media_t: float) -> int:
        """Convert a future media-time to a local nanosecond deadline."""
        with self._lock:
            if not self.playing or self.rate <= 0:
                return time.monotonic_ns()
            delta_media_s = media_t - self.anchor_media_t
            target_server_ns = self.anchor_server_ns + int(delta_media_s * 1e9 / self.rate)
            return target_server_ns - self.clock_offset_ns


# ============================ THE CLIENT ======================================
class HapticClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.tcp_sock: Optional[socket.socket] = None
        self.udp_sock: Optional[socket.socket] = None
        self.udp_port = 0

        self.timeline: Optional[dict] = None
        self.events: List[dict] = []
        self.event_times: List[float] = []

        self.clock = MediaClock()
        self.stop_event = threading.Event()
        self.threads: List[threading.Thread] = []

        # scheduler bookkeeping: which event indices we've already fired
        self._fired_up_to = -1           # last index already fired
        self._fired_lock = threading.Lock()

        # Clock-sync replies arrive through the regular TCP recv loop and get
        # routed here. (They can't be read inline, because the server also
        # sends `timeline` and `play` messages on the same socket — having the
        # main loop be the single recv owner avoids the race.)
        self._time_resp_q: "queue.Queue[dict]" = queue.Queue()

    # ---------- lifecycle ----------
    def connect(self) -> None:
        # UDP first so we know its port before we send hello
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.bind(("0.0.0.0", 0))
        self.udp_port = self.udp_sock.getsockname()[1]
        log.info("UDP listening on port %d", self.udp_port)

        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.connect((self.host, self.port))
        log.info("TCP connected to %s:%d", self.host, self.port)

        send_message(self.tcp_sock, {
            "msg": "hello",
            "client": "python-fake",
            "client_version": "0.1",
            "udp_port": self.udp_port,
            "capabilities": {
                # values a real Android client would report; here just stubbed
                "amplitude_control": True,
                "primitives": ["CLICK", "TICK", "LOW_TICK", "THUD"],
                "vibrator_api": 31,
            },
        })

    def start_loops(self) -> None:
        for target, name in [
            (self._tcp_recv_loop,  "tcp-recv"),
            (self._udp_recv_loop,  "udp-recv"),
            (self._scheduler_loop, "scheduler"),
        ]:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self) -> None:
        self.stop_event.set()
        for s in (self.tcp_sock, self.udp_sock):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass

    # ---------- clock sync ----------
    def clock_sync(self, samples: int = 8) -> None:
        """SNTP-style. Send `samples` time_req/time_resp pairs, take the best
        half by RTT, average their offsets. Requires the TCP recv loop to be
        running already (it routes time_resp messages into our queue)."""
        offsets = []
        for _ in range(samples):
            t0_local = time.monotonic_ns()
            send_message(self.tcp_sock, {"msg": "time_req",
                                          "t0_client_ns": t0_local})
            try:
                resp = self._time_resp_q.get(timeout=1.0)
            except queue.Empty:
                continue
            t1_local = time.monotonic_ns()
            if resp.get("msg") != "time_resp":
                continue
            rtt = t1_local - t0_local
            offset = resp["t_server_ns"] - (t0_local + rtt // 2)
            offsets.append((rtt, offset))
            time.sleep(0.01)
        if not offsets:
            log.warning("clock sync got no responses")
            return
        # discard the worst half
        offsets.sort(key=lambda x: x[0])
        keep = offsets[: max(1, len(offsets) // 2)]
        avg_offset = sum(o for _, o in keep) // len(keep)
        avg_rtt = sum(r for r, _ in keep) // len(keep)
        self.clock.set_offset(avg_offset)
        log.info("clock sync: offset=%+.3fms  rtt=%.3fms  (kept %d/%d samples)",
                 avg_offset / 1e6, avg_rtt / 1e6, len(keep), len(offsets))

    # ---------- TCP loop ----------
    def _tcp_recv_loop(self) -> None:
        while not self.stop_event.is_set():
            msg = recv_message(self.tcp_sock)
            if msg is None:
                log.info("TCP closed by server")
                self.stop_event.set()
                return
            kind = msg.get("msg")
            if kind == "timeline":
                self._on_timeline(msg["data"])
            elif kind == "time_resp":
                self._time_resp_q.put(msg)
            elif kind == "play":
                self._anchor_from(msg, playing=True)
                log.info("PLAY  media_t=%.3f  rate=%.2f",
                         msg["media_t"], msg.get("rate", 1.0))
            elif kind == "pause":
                self._anchor_from(msg, playing=False)
                log.info("PAUSE media_t=%.3f", msg["media_t"])
            elif kind == "seek":
                self._anchor_from(msg, playing=self.clock.playing)
                self._reset_scheduler_to(msg["media_t"])
                log.info("SEEK  media_t=%.3f", msg["media_t"])
            elif kind == "rate":
                self.clock.anchor(self.clock.media_t_now(),
                                  msg["t_server_ns"], msg["rate"],
                                  self.clock.playing)
                log.info("RATE  %.2f", msg["rate"])
            else:
                log.debug("tcp recv: %r", msg)

    def _on_timeline(self, tl: dict) -> None:
        self.timeline = tl
        self.events = sorted(tl.get("events", []), key=lambda e: e["time"])
        self.event_times = [e["time"] for e in self.events]
        log.info("timeline received: source=%s duration=%.1fs events=%d",
                 tl.get("source"), tl.get("duration", 0.0), len(self.events))
        # reset firing state — fresh timeline means fresh scheduling
        with self._fired_lock:
            self._fired_up_to = -1

    def _anchor_from(self, msg: dict, playing: bool) -> None:
        self.clock.anchor(msg["media_t"], msg["t_server_ns"],
                          msg.get("rate", self.clock.rate), playing)

    def _reset_scheduler_to(self, media_t: float) -> None:
        # mark all events whose time < media_t as already fired (we skip past
        # them); subsequent events will fire in due course
        with self._fired_lock:
            self._fired_up_to = bisect.bisect_right(self.event_times, media_t) - 1

    # ---------- UDP loop ----------
    def _udp_recv_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.udp_sock.settimeout(1.0)
                data, _ = self.udp_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                msg = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if msg.get("msg") == "sync":
                self.clock.anchor(msg["media_t"], msg["t_server_ns"],
                                  msg.get("rate", 1.0), playing=True)

    # ---------- scheduler loop ----------
    def _scheduler_loop(self) -> None:
        """Naive but adequate: every 10ms, check the upcoming N events and
        fire any whose deadline has passed. A production client would use a
        proper timer queue, but this is plenty for a demo."""
        while not self.stop_event.is_set():
            time.sleep(0.005)
            if not self.events or not self.clock.playing:
                continue
            t_now = self.clock.media_t_now()
            with self._fired_lock:
                # find events whose time <= t_now but index > _fired_up_to
                start = self._fired_up_to + 1
                idx = start
                while idx < len(self.events) and self.events[idx]["time"] <= t_now:
                    ev = self.events[idx]
                    target_local_ns = self.clock.local_ns_for_media_t(ev["time"])
                    fire_haptic(ev, time.monotonic_ns(), target_local_ns, "scheduled")
                    idx += 1
                self._fired_up_to = idx - 1

    # ---------- manual trigger (for the demo's interactive 'f' key) ----------
    def fire_manual(self) -> None:
        """Fire one haptic event manually — uses the event nearest to the
        current media-time. This is what a tester would press to verify the
        haptic playback path is wired up without waiting for an event in the
        timeline to scroll past."""
        if not self.events:
            print("  (no timeline yet, nothing to fire)")
            return
        t_now = self.clock.media_t_now()
        # nearest event
        ni = min(range(len(self.events)),
                 key=lambda i: abs(self.events[i]["time"] - t_now))
        ev = self.events[ni]
        now_ns = time.monotonic_ns()
        fire_haptic(ev, now_ns, now_ns, "manual")


# ============================ MAIN ============================================
def main():
    ap = argparse.ArgumentParser(
        description="Fake Android client for testing the haptic server")
    ap.add_argument("--server", default=None,
                    help="HOST:PORT to skip NSD discovery and connect directly")
    ap.add_argument("--discover-timeout", type=float, default=5.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.server:
        host, port = args.server.split(":")
        addr = (host, int(port))
    else:
        log.info("discovering via mDNS (%s) ...", SERVICE_TYPE)
        addr = discover(args.discover_timeout)
        if addr is None:
            log.error("no server found via mDNS; pass --server HOST:PORT")
            sys.exit(2)

    client = HapticClient(addr[0], addr[1])
    client.connect()
    client.start_loops()           # must be running before clock_sync (it
                                   # routes time_resp messages through a queue)
    client.clock_sync(samples=8)

    print()
    print("=" * 70)
    print("  Connected. Interactive keys:")
    print("    f <enter>   fire a manual haptic event (nearest to current t)")
    print("    q <enter>   quit")
    print("  Scheduled events from the timeline will fire as media-time")
    print("  advances. Each line below corresponds to one vibration command")
    print("  that a real Android client would deliver to the haptic engine.")
    print("=" * 70)
    print()

    try:
        while not client.stop_event.is_set():
            line = sys.stdin.readline()
            if not line:
                break
            cmd = line.strip().lower()
            if cmd == "q":
                break
            elif cmd == "f":
                client.fire_manual()
            elif cmd == "":
                continue
            else:
                print(f"  unknown command: {cmd!r}; use 'f' or 'q'")
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        log.info("client stopped")


if __name__ == "__main__":
    main()
