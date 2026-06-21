# Haptic Sports — Wire Protocol Specification

**Version 1** · pragmatic spec for implementers

This document defines exactly what goes on the wire between the **laptop
server** (Python; runs on the machine playing the video) and a **client**
(typically an Android phone running the haptic app). It is the authoritative
reference when the architecture docs and the protocol disagree.

The reader is implementing one side and needs to interoperate with the other.
Companion documents:

- `HAPTIC_ANDROID_ARCHITECTURE.md` — how to structure the Android app (the *why*)
- `HAPTIC_SERVER_ARCHITECTURE.md` — how the laptop server integrates with the player
- `haptic_server.py`, `haptic_client_demo.py` — the reference implementation

If you're an LLM implementing the Android client, this is the document to
return to whenever you write a line of networking code. Everything else
explains *why* the protocol is shaped this way; this explains *what* the
bytes are.

---

## 1. At a glance

```
   ┌──────────── LAPTOP (server) ────────────┐
   │   advertises:  _haptics._tcp.local.     │
   │   TCP listens on port  47821            │
   │   UDP listens on port  47822            │
   └─────────────────────────────────────────┘
                       ▲ ▼            ▲
              TCP control            UDP sync
              (reliable, ordered)    (best-effort, fast)
                       ▲ ▼            ▲
   ┌──────────── PHONE (client) ─────────────┐
   │   browses for _haptics._tcp.local.      │
   │   opens random UDP port                 │
   │   tells server that port in `hello`     │
   └─────────────────────────────────────────┘
```

Three channels:

| Channel    | Transport | Direction       | What flows on it                          |
|------------|-----------|-----------------|--------------------------------------------|
| Discovery  | mDNS      | server → all    | service advertisement (once)               |
| Control    | TCP       | both ways       | hello, capabilities, timeline, clock-sync, play/pause/seek/rate |
| Sync       | UDP       | server → client | `sync` pulses ~5–10 Hz during playback     |

---

## 2. Discovery (mDNS / Bonjour / NSD)

The laptop registers exactly one mDNS service:

- **Service type:**  `_haptics._tcp.local.` (note the trailing dot)
- **Default port:**  47821 (TCP)
- **Service name:**  arbitrary, default `haptic-laptop`
- **TXT properties:**
  - `udp_port` — the UDP port the laptop is *listening on* for sync replies
    (currently unused by the client but reserved for future expansion;
    the client tells the server its own UDP port via `hello`).
  - `version` — protocol version, currently `"1"`

The client browses for `_haptics._tcp.local.`, resolves the first matching
service to get host + TCP port, and connects via TCP.

On Android:
```kotlin
nsdManager.discoverServices("_haptics._tcp", NsdManager.PROTOCOL_DNS_SD, listener)
```
Note: Android's `NsdManager` takes the service type **without** the
`.local.` suffix (it's added internally by mDNS).

---

## 3. Message framing (TCP control channel)

Every message on the TCP channel — both directions — uses the same frame:

```
 ┌─────────────────────┬───────────────────────────────────────────┐
 │  4-byte length (N)  │  N bytes of UTF-8 JSON                    │
 │  big-endian uint32  │  (no trailing newline, no null terminator)│
 └─────────────────────┴───────────────────────────────────────────┘

 Byte 0   Byte 1   Byte 2   Byte 3   Byte 4 .. Byte (3+N)
 [---length, big-endian---][----------JSON body----------]

 Example: a 7-byte JSON body {"x":1} would be framed as:
 0x00 0x00 0x00 0x07 0x7B 0x22 0x78 0x22 0x3A 0x31 0x7D
 │     length = 7     │  {    "    x    "    :    1    }
```

Constraints:

- **Length** is in **bytes**, not characters. UTF-8 multi-byte sequences
  count by their byte length.
- **Maximum frame size**: the reference server rejects messages with
  `length > 64 MiB`. Timelines for a full match are well under 1 MiB; this
  cap exists just to defend against garbage on the socket.
- **No streaming partial frames**: every read must produce one complete
  message before processing.
- **JSON is compact** (`json.dumps(..., separators=(",", ":"))`): no
  whitespace between tokens. Parsers MUST tolerate both compact and
  pretty-printed JSON, but produce compact.

### Implementation pattern (Kotlin sketch)

```kotlin
// Reading one message:
val header = ByteArray(4); readFully(header)
val n = ByteBuffer.wrap(header).int  // big-endian by default in ByteBuffer
require(n in 1..(64 * 1024 * 1024))
val body = ByteArray(n); readFully(body)
val msg: JsonObject = parseJson(body.toString(Charsets.UTF_8))

// Writing:
val payload = msg.toString().toByteArray(Charsets.UTF_8)
out.write(ByteBuffer.allocate(4).putInt(payload.size).array())
out.write(payload)
out.flush()
```

### A real gotcha that bit the reference implementation

If your code reads from the TCP socket from **two places** (e.g. `clock_sync`
reading inline while a recv loop reads in parallel), you get a race: the
clock-sync reader can consume a `timeline` message intended for the recv
loop. **The recv loop must be the single owner of TCP reads from start to
finish.** Route clock-sync replies through an in-memory queue, not by
reading the socket directly.

---

## 4. Message catalog (TCP control)

All messages are JSON objects with a `"msg"` field naming the kind.

### 4.1 `hello` — client → server (sent once on connect)

The first message the client sends. Identifies the client, declares its
capabilities, and tells the server the UDP port to send sync pulses to.

```json
{
  "msg": "hello",
  "client": "android-haptic",
  "client_version": "0.1",
  "udp_port": 41665,
  "capabilities": {
    "amplitude_control": true,
    "primitives": ["CLICK", "TICK", "LOW_TICK", "THUD"],
    "vibrator_api": 31
  }
}
```

Fields:

| Field                          | Type    | Required | Notes                              |
|--------------------------------|---------|----------|------------------------------------|
| `msg`                          | string  | yes      | always `"hello"`                   |
| `client`                       | string  | yes      | client id, free-form (`"android-haptic"`, `"python-fake"`, …) |
| `client_version`               | string  | recommended | semver-ish, free-form         |
| `udp_port`                     | int     | yes      | port the client's UDP socket is *listening on* |
| `capabilities`                 | object  | yes      | the client's haptic abilities      |
| `capabilities.amplitude_control` | bool   | yes      | `Vibrator.hasAmplitudeControl()`   |
| `capabilities.primitives`      | array   | yes      | primitive names supported (see below) |
| `capabilities.vibrator_api`    | int     | yes      | `Build.VERSION.SDK_INT`            |

Primitive name strings (these correspond to Android `VibrationEffect.Composition` constants but are sent as plain strings so the protocol stays platform-neutral):

`CLICK`, `TICK`, `LOW_TICK`, `THUD`, `QUICK_RISE`, `SLOW_RISE`, `QUICK_FALL`, `SPIN`

A client that supports none of these (older API) sends an empty array.

**Server response:** the server replies immediately with a `timeline`
message (if it has one) and — if playback is in progress — a `play` message
anchored to the current extrapolated media-time. The client should be
prepared to receive these *before* it has sent anything else.

### 4.2 `timeline` — server → client

The complete haptic timeline for the currently loaded video. Sent right
after `hello`, and again any time the laptop user loads a different video.

```json
{
  "msg": "timeline",
  "data": { ... timeline JSON, see §6 ... }
}
```

On receipt, the client SHOULD:

1. Replace any previous timeline atomically.
2. Sort `events` by `time` if not already sorted.
3. Build whatever index it uses for "next event after t" lookups.
4. Reset its "already-fired-up-to" pointer.

### 4.3 `time_req` / `time_resp` — clock sync

Pair of messages used to estimate the offset between the client's monotonic
clock and the server's monotonic clock. Run **once after `hello`**, and
optionally again later if the offset drifts.

Client → server:
```json
{ "msg": "time_req", "t0_client_ns": 123456789000 }
```
- `t0_client_ns` is the client's `System.nanoTime()` (or equivalent) **at
  the moment of sending**. The server echoes this verbatim — never modify
  it on either side.

Server → client (replies as fast as possible):
```json
{ "msg": "time_resp", "t0_client_ns": 123456789000, "t_server_ns": 987654321000 }
```
- `t0_client_ns` is the original value, untouched.
- `t_server_ns` is the server's monotonic clock at the moment of receipt.

Client computes, on receive:
```
t1_client_ns = System.nanoTime()         // local clock at receive
rtt_ns       = t1_client_ns - t0_client_ns
offset_ns    = t_server_ns - (t0_client_ns + rtt_ns / 2)
```

Repeat 5–10 times in quick succession. Discard the half with the *worst*
(highest) RTT. Average the offsets of the kept samples.

The result is stored as `clock_offset_ns = server_monotonic - client_monotonic`.
At any later moment, the client computes `now_server_ns = System.nanoTime() + clock_offset_ns`.

### 4.4 `play` — server → client

Playback started, or playback in progress was re-anchored.

```json
{
  "msg": "play",
  "media_t": 0.000,
  "rate": 1.0,
  "t_server_ns": 987654321000
}
```

| Field         | Type   | Notes                                                          |
|---------------|--------|----------------------------------------------------------------|
| `media_t`     | float  | media-time (seconds) at which playback is anchored             |
| `rate`        | float  | playback rate (1.0 = normal, 0.5 = half-speed, etc.)           |
| `t_server_ns` | int    | server's monotonic-clock nanosecond timestamp at anchor moment |

The client sets:
```
anchor_media_t   = media_t
anchor_server_ns = t_server_ns
rate             = rate
playing          = true
```

A `play` with `media_t = anchor_media_t` and a fresh `t_server_ns` is also
how the server re-anchors after a `seek`.

### 4.5 `pause` — server → client

```json
{ "msg": "pause", "media_t": 12.345, "t_server_ns": 987654321000 }
```

The client sets `playing = false` and stops advancing media-time. Any
already-scheduled vibrations whose deadline has not yet arrived SHOULD be
cancelled.

### 4.6 `seek` — server → client

```json
{ "msg": "seek", "media_t": 42.000, "t_server_ns": 987654321000 }
```

Playback jumped to a new media-time. The client SHOULD:

1. Cancel any pending scheduled vibrations.
2. Re-anchor: `anchor_media_t = media_t`, `anchor_server_ns = t_server_ns`.
3. Resume scheduling forward from the new position (keeping the previous
   `playing` and `rate` state).

### 4.7 `rate` — server → client

Playback rate changed (slow-mo or fast playback).

```json
{ "msg": "rate", "rate": 0.5, "t_server_ns": 987654321000 }
```

Before applying the new rate, the client SHOULD re-anchor using the
*current* extrapolated media-time at the *old* rate, so the media-clock
math stays continuous. In practice:

```kotlin
val currentMediaT = mediaClock.mediaTimeNow()
mediaClock.anchor(media_t = currentMediaT,
                  server_ns = msg.t_server_ns,
                  rate = msg.rate,
                  playing = true)
```

Pending scheduled vibrations should be re-scheduled with the new rate.

### 4.8 Unknown messages

Both sides MUST ignore messages with an unknown `"msg"` value. Logging at
debug level is fine. This lets the protocol be extended without breaking
older clients.

---

## 5. UDP sync channel

Sync pulses are JSON-over-UDP datagrams sent by the server to each client's
`udp_port` (from that client's `hello`) at roughly 5–10 Hz **during
playback only**. They're how the client corrects drift between sync points.

### 5.1 Datagram layout

UDP datagrams have **no length prefix** — each datagram is one complete
message:

```
 ┌─────────────────────────────────────────────────┐
 │  raw UDP datagram payload                       │
 │  = one UTF-8 JSON object                        │
 │  e.g.  {"msg":"sync","media_t":3.125,           │
 │         "t_server_ns":987654321000,"rate":1.0}  │
 └─────────────────────────────────────────────────┘
```

Recommended `recvfrom` buffer size: **2048 bytes**. Sync messages are tiny
(under 200 bytes in practice) but the buffer gives slack.

### 5.2 `sync` message

```json
{
  "msg": "sync",
  "media_t": 3.125,
  "t_server_ns": 987654321000,
  "rate": 1.0
}
```

Identical fields to `play`, sent on a periodic timer rather than as a state
change. On receipt, the client re-anchors its media-clock just as it would
for `play`:

```kotlin
mediaClock.anchor(media_t = msg.media_t,
                  server_ns = msg.t_server_ns,
                  rate = msg.rate,
                  playing = true)
```

### 5.3 What if sync pulses stop arriving?

Sync is best-effort. The client uses its local clock between pulses, so a
few dropped UDP packets cost zero events.

If pulses stop for **more than ~2 seconds** during what the client believes
is playback, the client MAY assume the server has paused, disconnected, or
the network has dropped — but this is an inference, not authoritative. The
authoritative signals are the TCP `pause` and the TCP socket itself closing.

### 5.4 No UDP messages flow client → server

The client only **listens** on UDP. It never sends UDP to the server.

---

## 6. Timeline data format

The `data` field in the `timeline` message is a JSON object. The
authoritative schema:

```jsonc
{
  "version": 2,        // schema version (integer)
  "source": "match.mp4",  // original media filename (informational)
  "duration": 412.5,   // total media duration in seconds (float)

  "events": [
    {
      "time": 12.480,        // media-time in SECONDS (float, required)
      "intensity": 0.82,     // 0.0..1.0 vibration strength (float, required)
      "type": "hit",         // always "hit" — see §6.1 (string, required)
      "vision_type": "strike" // refined classification — see §6.2 (string|null, optional)
    },
    ...
  ]
}
```

> **Minimal payload:** the server sends only the four fields above per event.
> All research fields (`db`, `hf_ratio`, `centroid`, `flatness`, etc.) and
> all annotation/analyzer metadata (`calibration`, `params`, `updated_at`,
> `annotation_log`, etc.) are stripped before transmission. The server uses
> an allowlist — any new field added in future is excluded by default unless
> explicitly opted in.

> **Pre-filtered:** the server applies all suppression rules (VAD speech
> detection, vision confirmation) before sending. The client receives only
> events that should produce haptic feedback — it does not need to filter.

### 6.1 Field semantics — what the client MUST honor

- **`time`** is in seconds (not milliseconds). Convert if your internal
  representation uses ms.
- **`intensity`** is the calibrated 0.0–1.0 vibration strength for **this
  video**. It is **not** comparable across videos. Do not attempt to
  normalize, scale, or "correct" it across recordings.
- **`type`** is always `"hit"` in the current pipeline. Audio onset detection
  produces all events; `vision_type` (§6.2) carries the refined classification.
  **Unknown `type` values MUST NOT crash the client.** Fall back to a default
  haptic pattern for any unrecognised value.

### 6.2 Field semantics — what the client MAY use

- **`vision_type`** is the vision-refined event classification: `"strike"`,
  `"bounce"`, or `null` (undetermined / no vision data). **This is the field
  to switch on for haptic pattern selection** — not `type`. When present and
  non-null, use it to choose a pattern (e.g. a sharp click for a racket
  strike, a thud for a bounce). When null or absent, fall back to a generic
  hit pattern.

  Recommended Kotlin pattern:
  ```kotlin
  val pattern = when (event.visionType) {
      "strike" -> HapticPattern.STRIKE
      "bounce" -> HapticPattern.BOUNCE
      else     -> HapticPattern.HIT   // null, absent, or unknown
  }
  ```

### 6.3 Android implementation notes

> **If your Android client was written against an earlier version of this
> document**, check the following:
>
> - **`type` vs `vision_type`**: older versions of this doc described `type`
>   as `"strike"` or `"bounce"`. In practice `type` has always been `"hit"`.
>   If your code switches on `type` for haptic pattern selection, change it
>   to switch on `vision_type` instead.
> - **Removed fields**: `db`, `hf_ratio`, `centroid`, `calibration`,
>   `params`, `sample_rate_analyzed` are no longer sent. Grep your client
>   source for these strings; any reference to them can be removed.
> - **Unknown fields**: if your JSON parser is strict (rejects unknown
>   fields), make it lenient — the protocol can add fields in future.

### 6.4 Unknown fields

Future protocol versions may add new fields. **Clients MUST silently ignore
any field they do not recognise.** This allows the protocol to evolve
without breaking existing clients.

### 6.4 Sorting

The reference server sorts events by `time` before sending. The client
SHOULD still sort defensively on receipt, since the order is not part of
the contract.

---

## 7. Example session: full connect → play → disconnect

A complete trace of a successful session. Times are wall-clock to show
ordering; field values are illustrative.

```
T+0.000  CLIENT  starts mDNS browse for _haptics._tcp.local.
T+0.030  CLIENT  receives ServiceFound: haptic-laptop at 192.168.1.42:47821
T+0.031  CLIENT  resolves service, gets InetAddress + port
T+0.035  CLIENT  opens UDP socket, bind to 0.0.0.0:0 (OS picks port 41665)
T+0.036  CLIENT  opens TCP socket, connect to 192.168.1.42:47821
T+0.040  CLIENT  starts TCP recv loop on background thread
T+0.041  CLIENT  TX TCP -> {"msg":"hello","client":"android-haptic",
                            "client_version":"0.1","udp_port":41665,
                            "capabilities":{"amplitude_control":true,
                              "primitives":["CLICK","TICK","LOW_TICK","THUD"],
                              "vibrator_api":31}}

T+0.043  SERVER  RX hello, records caps + udp_addr = (192.168.1.99, 41665)
T+0.044  SERVER  TX TCP -> {"msg":"timeline","data":{ ...8 KB... }}
T+0.045  SERVER  (playback is in progress; also sends current play anchor)
T+0.045  SERVER  TX TCP -> {"msg":"play","media_t":12.300,"rate":1.0,
                            "t_server_ns":987654300000000}

T+0.052  CLIENT  RX timeline    -> parses 213 events, indexes by time
T+0.052  CLIENT  RX play        -> anchors media-clock: media_t=12.300

# --- Clock sync (8 samples) ---
T+0.060  CLIENT  TX -> {"msg":"time_req","t0_client_ns":1234567890000}
T+0.061  SERVER  TX -> {"msg":"time_resp","t0_client_ns":1234567890000,
                        "t_server_ns":987654308000000}
T+0.062  CLIENT  RX time_resp   -> sample 1: rtt=1.8ms, offset=…
                                   (route via time_resp queue, not direct read)
... (repeats 7 more times in ~80 ms total) ...
T+0.140  CLIENT  clock_sync complete: offset=-0.42ms (median of best 4 samples)

# --- Steady-state playback ---
T+0.250  SERVER  TX UDP -> {"msg":"sync","media_t":12.515,
                            "t_server_ns":987654515000000,"rate":1.0}
T+0.251  CLIENT  RX sync        -> re-anchors media-clock
T+0.375  SERVER  TX UDP -> {"msg":"sync","media_t":12.640,...}
T+0.500  SERVER  TX UDP -> {"msg":"sync","media_t":12.765,...}
                                                   ...(continuing at 8 Hz)
T+1.230  CLIENT  scheduler: event #43 (time=13.480, type=hit, vision_type=strike, intensity=0.82)
                 deadline reached -> vibrator.vibrate(...)

# --- User pauses video on laptop ---
T+3.140  SERVER  TX TCP -> {"msg":"pause","media_t":15.180,
                            "t_server_ns":987658130000000}
T+3.142  CLIENT  RX pause       -> playing=false, cancels scheduled vibrations
                                   (no more sync pulses arrive until play resumes)

# --- User seeks forward ---
T+4.000  SERVER  TX TCP -> {"msg":"seek","media_t":42.000,...}
T+4.002  CLIENT  RX seek        -> re-anchor, reset fired-up-to pointer

# --- User resumes ---
T+4.100  SERVER  TX TCP -> {"msg":"play","media_t":42.000,"rate":1.0,...}
T+4.102  CLIENT  RX play        -> playing=true; sync pulses resume

# --- Disconnect ---
T+99.0   USER    closes the app on the phone
T+99.0   CLIENT  closes TCP socket and UDP socket
T+99.0   SERVER  RX socket close (recv returns 0) -> drops the client
                                                     (UDP target removed)
```

The thing to internalize from this trace: the **client's TCP recv loop is
already running when the server sends `timeline` and `play` immediately
after receiving `hello`**. If it isn't — if the client tries to read those
messages with a one-off `recv()` instead of through the loop — they're lost
or misinterpreted. This is the single most important sequencing rule.

---

## 8. Errors and edge cases

### 8.1 Malformed frames

The reference server rejects messages whose declared length is zero or
exceeds 64 MiB. The reference client doesn't enforce a cap (LAN-only,
trusted server) but a hostile client could be defended against by adding
the same cap.

A JSON parse failure on a received frame: log it and **continue reading**.
Don't close the socket on one bad frame — the other side may be fine.

### 8.2 TCP socket closes unexpectedly

Treated as disconnection. The client returns to mDNS discovery. The server
removes the client from its broadcast list.

### 8.3 UDP datagram with unknown `msg` field

Ignore silently. Don't log at higher than debug level — UDP can receive
stray packets, especially during testing.

### 8.4 `hello` not received

The server does not time out clients that connect but don't send `hello`.
However, until `hello` arrives, the server can't send sync pulses (no UDP
address) — playback events are silently lost on that client. Send `hello`
immediately after `connect()`.

### 8.5 Receiving a `timeline` mid-playback

Treat as an atomic replacement. Clear scheduled events from the old
timeline, build the new index, resume scheduling from the current
media-time against the new timeline.

### 8.6 Clock drift during long sessions

For sessions over ~5 minutes on devices with thermal clock skew, consider
re-running clock-sync once. The reference client does not currently do
this. Sync pulses already correct most drift, so it's only relevant if
sync pulses are unreliable.

---

## 9. Versioning

The protocol version is currently **1**, advertised in the mDNS TXT record.
The reference implementation does not yet check version compatibility on
the client side — it should, before the protocol changes.

Breaking changes will bump the protocol version. Non-breaking additions
(new message kinds, new optional fields) will not.

---

## 10. Quick reference card

| Item                       | Value                                |
|----------------------------|--------------------------------------|
| mDNS service type          | `_haptics._tcp.local.`               |
| Default TCP port           | 47821                                |
| Default UDP port (server)  | 47822                                |
| Client UDP port            | random, declared in `hello`          |
| TCP frame format           | 4-byte BE length + UTF-8 JSON body   |
| Max TCP frame size         | 64 MiB                               |
| UDP datagram format        | UTF-8 JSON, no length prefix         |
| UDP buffer size            | 2048 bytes                           |
| Time units on the wire     | media-time in **seconds** (float)    |
|                            | server clock in **nanoseconds** (int)|
| TCP message kinds          | hello, time_req, time_resp, timeline, play, pause, seek, rate |
| UDP message kinds          | sync                                 |
| Sync pulse rate            | 5–10 Hz (server's choice)            |

---

**End of protocol specification.** When in doubt, the reference implementation
(`haptic_server.py`, `haptic_client_demo.py`) is the ground truth. If this
document disagrees with their behavior, the code is right and the doc is
wrong — please report it so the doc can be fixed.
