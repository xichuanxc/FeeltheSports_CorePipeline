# Haptic Sports Server (Python) — Architecture & Integration

This document describes the **laptop-side server** that broadcasts haptic
timelines and sync pulses to a connected client (the Android phone, or the
Python demo client used for protocol testing).

Companion document: `HAPTIC_ANDROID_ARCHITECTURE.md`, which specifies the
phone client. **Read that first** — the protocol is defined there, and this
doc is the server's view of the same protocol. This doc focuses on the
server's API, threading model, and integration with the PySide6 video player.

The reader is either you (as an integrator), or an LLM coding assistant
extending the server.

---

## 1. The two-file architecture

The server side ships as two files:

```
server.py               the server module (importable + runnable)
client_demo.py          a fake Android client for testing (NSD discovery,
                        TCP/UDP protocol, manual trigger key)
```

`server.py` is **decoupled** from the PySide6 player on purpose. The
player imports `HapticServer` and calls a handful of `publish_*` methods.
Nothing in the server depends on Qt, on `librosa`, or on the player's GUI
loop. This separation matters for three reasons:

- **Testability.** You can run the server with no GUI and a hardcoded timeline.
- **Latency.** The server runs sockets on its own threads, never touching the
  Qt event loop, so haptic timing doesn't fight with painting and video
  rendering.
- **Reuse.** A future iOS or web client uses the same server unchanged.

The demo client is similarly self-contained — it exists only to verify the
protocol works end-to-end without needing a real phone.

---

## 2. The protocol in one paragraph

The protocol is documented in detail in `HAPTIC_ANDROID_ARCHITECTURE.md` §4.
In summary: the server advertises `_haptics._tcp.local.` via mDNS; clients
discover and open a TCP socket; both sides exchange length-prefixed JSON
messages (4-byte big-endian length followed by UTF-8 JSON). The client sends
`hello` with its capabilities and UDP port; the server replies with the
current `timeline` (and, if playback is in progress, an immediate `play`
anchor). Then SNTP-style clock-sync runs over TCP (`time_req`/`time_resp`).
During playback, the server broadcasts lightweight `sync` messages over UDP
at 5–10 Hz; control messages (`play`, `pause`, `seek`, `rate`) go over TCP.

---

## 3. Server module — public API

```python
from haptic_server import HapticServer

server = HapticServer(
    timeline_dict=loaded_json,    # optional; can be set later via set_timeline
    tcp_port=47821,               # defaults shown
    udp_port=47822,
    service_name="haptic-laptop", # mDNS service name
)

server.start()                    # advertise + begin accepting clients

# Player-side state events:
server.set_timeline(tl_dict)      # update timeline (resent to current clients)
server.publish_play(media_t, rate=1.0)
server.publish_pause(media_t)
server.publish_seek(media_t)
server.publish_rate(rate)
server.publish_sync(media_t)      # call ~5-10 Hz during playback

server.client_count               # property: live clients

server.stop()                     # clean shutdown (call on app quit)
```

### Behavior contract

- All `publish_*` methods are **thread-safe**. Call them from any thread,
  including the Qt main thread.
- All `publish_*` methods are **non-blocking** in practice (socket sends
  with `sendall` could in theory block briefly under back-pressure, but on
  LAN this is microseconds).
- `start()` returns only after sockets are bound and mDNS is registered, so
  the caller can immediately announce the service to a user.
- `stop()` is idempotent and safe to call from any thread or signal handler.
- New clients connecting mid-playback are automatically caught up: the server
  re-sends the current timeline and an extrapolated `play` anchor at the
  current media-time on `hello`. The integrator does **not** need to call
  `publish_play` again when a late client connects.
- `set_timeline` resends the timeline to all currently connected clients;
  this is what to call when the user switches to a different video.

### What the server does NOT do

- Does not advance media-time on its own. The player tells it the current
  `media_t` on every `publish_sync` call. (Exception: when extrapolating for
  a late-connecting client, the server uses its anchor + monotonic clock to
  estimate the current media-time — but this never *replaces* the player's
  authoritative value.)
- Does not call back into the player. Information flows player → server only.
- Does not validate the timeline JSON. The player is trusted to pass valid
  data conforming to the schema in `HAPTIC_ANDROID_ARCHITECTURE.md` §3.
- Does not implement any haptic logic itself. It distributes; it does not
  decide what to feel.

---

## 4. Threading model

```
┌────────────────────────────────────────────────────────────────────┐
│  HapticServer (main thread, owns lifecycle)                        │
│                                                                    │
│  ┌─ accept-loop thread ──────────────────────────────────────────┐ │
│  │  TCP server socket .accept() in a loop                        │ │
│  │  spawns a client-loop thread per accepted connection          │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌─ client-loop thread (one per client) ─────────────────────────┐ │
│  │  recv length-prefixed JSON forever                            │ │
│  │  handles `hello` (saves caps + UDP port + sends timeline+play)│ │
│  │  handles `time_req` (replies with `time_resp`)                │ │
│  │  exits & drops client on socket close                         │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  (publish_* called from caller's thread; iterates clients,         │
│   sends through each client's per-socket write_lock; UDP sends     │
│   are lock-free since UDP has no in-order constraint)              │
└────────────────────────────────────────────────────────────────────┘
```

Concurrency invariants worth knowing:
- The list of clients (`self._clients`) is protected by `self._clients_lock`.
  Iterators take a snapshot under the lock, then operate outside it.
- Each `ClientState` has its own `write_lock` for serializing sends to that
  one socket. A slow client can't block sends to other clients.
- The timeline reference is protected by `self._timeline_lock`. (Only the
  reference; the dict itself is never mutated after assignment.)

---

## 5. Integration with the PySide6 player

The player needs five hooks. Each is a one-liner.

```python
# in PlayerWindow.__init__, after the timeline JSON is loaded:
self.server = HapticServer(timeline_dict=self.data, service_name="haptic-laptop")
self.server.start()

# wire up a sync-pulse timer (Qt timer in the player's main thread):
self.sync_timer = QTimer(self)
self.sync_timer.setInterval(125)             # 8 Hz; pick from 100–200 ms
self.sync_timer.timeout.connect(self._emit_sync_if_playing)

def _emit_sync_if_playing(self):
    if not self.is_paused:
        self.server.publish_sync(self.player.position() / 1000.0)

# in the play/pause/seek key handlers:
def _on_play(self):
    self.player.play()
    self.is_paused = False
    self.server.publish_play(self.player.position() / 1000.0,
                              rate=SPEED_STEPS[self.speed_index])
    self.sync_timer.start()

def _on_pause(self):
    self.player.pause()
    self.is_paused = True
    self.sync_timer.stop()
    self.server.publish_pause(self.player.position() / 1000.0)

def _on_seek(self, new_ms):
    self.player.setPosition(new_ms)
    self.server.publish_seek(new_ms / 1000.0)

def _on_rate_change(self):
    rate = SPEED_STEPS[self.speed_index]
    self.player.setPlaybackRate(rate)
    self.server.publish_rate(rate)

# on shutdown:
def closeEvent(self, e):
    self.server.stop()
    super().closeEvent(e)
```

A few honest notes about this integration:

- **Sync rate.** 5–10 Hz is the sweet spot. Higher costs bandwidth and CPU
  with no benefit (the phone's local media-clock fills the gaps); lower
  risks visible drift between sync pulses on slow playback. 8 Hz (125 ms) is
  what I'd default to.
- **Where `position()` comes from.** In a `QMediaPlayer`-based player this is
  the audio clock, in milliseconds. Convert to seconds (`/ 1000.0`) before
  passing to `publish_*` — the wire format is in seconds.
- **No need to gate publish_* on having clients.** All methods are no-ops
  when no clients are connected. Cleaner than the player keeping track.

---

## 6. Standalone mode (no player)

```
python server.py path/to/match.haptic.json [--simulate]
                                                  [--rate-hz N]
                                                  [--tcp-port N]
                                                  [--udp-port N]
                                                  [--name NAME]
```

`--simulate` is the key flag: the server pretends to play through the
timeline by advancing a virtual playhead in real time and emitting `sync`
pulses at `--rate-hz` (default 8). It loops at the end. Use this to develop
and test the network layer (and the Android client) without involving the
real Qt player at all.

Without `--simulate`, the server idles: clients can connect and receive the
timeline, but no events will fire because nothing is publishing `play`.
Useful for testing connect/disconnect logic in isolation.

---

## 7. The demo client (`client_demo.py`)

A Python fake-Android client. It exists to verify the server's protocol
without needing a real Android device, and as a reference implementation
for what the Android client will end up doing.

What it does:

1. Discovers the server via mDNS (or with `--server HOST:PORT` skips it).
2. Opens UDP first (to know its port), then TCP.
3. Sends `hello` with its UDP port and a stub capabilities dict.
4. Starts a TCP recv loop (importantly, before clock sync — see §8).
5. Runs SNTP-style clock sync (8 samples, keep best half by RTT).
6. Listens for UDP sync pulses; re-anchors its `MediaClock` from each.
7. Schedules and "fires" haptic events as their media-time arrives,
   printing one human-readable line per event with:
   - the event's media-time and type
   - intensity (numeric + a `#` bar)
   - **timing error** in milliseconds (actual fire time minus target)

Each printed line corresponds 1:1 to one `vibrator.vibrate(...)` call a real
Android client would make. The timing-error column is the diagnostic that
tells you whether the clock-sync and scheduling are working: with a healthy
LAN you should see single-digit-millisecond errors.

### Interactive keys

While the demo is connected and running:
- `f` + Enter — fire one **manual** haptic event using the nearest timeline
  event to the current media-time. Useful to verify the haptic playback
  path is wired without waiting for the timeline to scroll past one.
- `q` + Enter — quit.

Manual fires are clearly marked `[MANUAL]` in the output, scheduled ones
`[scheduled]`, so you can tell them apart.

### Running the demo

In one terminal:
```
python server.py match.haptic.json --simulate
```

In another:
```
python client_demo.py
# or, skipping mDNS discovery:
python client_demo.py --server 127.0.0.1:47821
```

A successful run looks like:
```
clock sync: offset=-0.012ms  rtt=0.06ms  (kept 4/8 samples)
  HAPTIC [scheduled] t=  1.000s  type=strike   intensity=0.50 ##########  timing-error= +1.9ms
  HAPTIC [scheduled] t=  1.500s  type=strike   intensity=0.90 ##########  timing-error= +2.1ms
  HAPTIC [MANUAL]    t=  2.200s  type=bounce   intensity=0.30 ######      timing-error= +0.0ms
  ...
```

---

## 8. Design gotchas worth knowing

### Why the demo client starts its recv loop before clock sync

Originally `clock_sync` read time_resp messages directly from the socket.
This broke as soon as the server started sending `timeline` and `play` on
the same socket — those messages would be consumed by `clock_sync` and
silently dropped. **The fix and the right pattern:** start the TCP recv
loop first, have it route `time_resp` messages into a `queue.Queue`, and
have `clock_sync` pull from that queue. This makes the recv loop the
*single owner* of the TCP read, which is the only safe pattern. The demo
client now does this; the Android client must do the same.

### The server's "send timeline on hello" replay

The server sends the timeline (and a play anchor, if playing) on `hello`,
not after any explicit client request. This is because:
- The client always needs the timeline immediately on connect — there's no
  meaningful "I don't want the timeline yet" state.
- Late-arriving clients during playback need to catch up to the current
  media-time, which the server can extrapolate from its anchor.

This works fine *as long as the client's recv loop is running* when the
server sends — which is why the previous note matters.

### Why sync is UDP, not TCP

TCP would serialize sync pulses behind any large in-flight message (e.g. a
new timeline being sent). UDP has no head-of-line blocking — sync pulses
get out instantly even if a 100 KB timeline message is in flight. And for
sync specifically, *late delivery is worse than no delivery*: a 200ms-old
sync pulse would mis-anchor the client. UDP's "drop if delayed" is the
correct semantics here. The cost — occasional dropped pulses — is fine
because the client coasts on its local clock between any two pulses.

### Per-client write locks, not a single server lock

Sending to one slow client must never block sends to other clients. Each
`ClientState` has its own `write_lock`, so a stalled phone can't freeze the
whole server. (For the same reason, `_broadcast_tcp` snapshots the client
list under the clients-lock, then sends *outside* it.)

### No backpressure on UDP

UDP sends are fire-and-forget. If a client's network is overwhelmed and
packets drop, the server doesn't know and doesn't care. The client recovers
from missing pulses by using its local clock. This is the right behavior;
adding "reliable UDP" would defeat the point of using UDP in the first place.

### Don't call publish_sync from the server's own thread

The integration pattern uses a Qt timer in the player thread. Don't be
tempted to have the server spawn its own "sync ticker" thread. The server
doesn't know whether playback is happening; the player does. Keep the
authority where it belongs.

---

## 9. Known limits and non-goals

- **Single laptop, multiple phones:** supported, untested. The server accepts
  any number of clients; broadcast logic iterates all. In practice expect to
  use one phone at a time.
- **Cross-network:** not supported. mDNS doesn't cross routers, and the
  whole design assumes the same LAN. Don't try to use this over the internet.
- **Authentication / encryption:** none. This is local-LAN only and not
  intended for hostile networks.
- **TLS / wss:** not implemented. The protocol is plain JSON over TCP/UDP.
- **Reconnect on the same TCP socket:** if the client drops, the server
  forgets it; the client must rediscover and reconnect. This is the simplest
  thing that works and is fine for the use case.
- **Variable timeline support:** the timeline is replaced atomically via
  `set_timeline`. Mid-playback timeline edits would require additional
  protocol design (event diffs) and aren't supported.

---

## 10. Quick reference

**Files:**
- `server.py` — server module (~330 lines, no Qt)
- `client_demo.py` — test client (~290 lines)

**Integration points in the player (5 calls):**
- `HapticServer(timeline_dict=...).start()` on launch
- `publish_play(t, rate)` / `publish_pause(t)` / `publish_seek(t)` / `publish_rate(r)` on state changes
- `publish_sync(t)` on a ~8 Hz timer during playback
- `set_timeline(tl)` when changing videos
- `stop()` on shutdown

**Standalone test loop:**
```
# Terminal 1:
python server.py match.haptic.json --simulate

# Terminal 2:
python client_demo.py
# press 'f' to fire a manual event, 'q' to quit
```

**Verified end-to-end:** NSD discovery, TCP handshake, timeline transfer,
clock sync (sub-millisecond on localhost LAN), UDP sync pulses, scheduled
event firing within ±5 ms of the target media-time on a quiet network.

---

**End of document.** Pair with `HAPTIC_ANDROID_ARCHITECTURE.md` for the
client-side perspective. Where the two docs disagree, the wire-format
description in the Android doc is authoritative.
