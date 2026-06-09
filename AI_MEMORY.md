# Sterling — AI Memory File
> Read this at the start of every session. Update only on meaningful changes (new files, decisions, direction shifts). Keep it tight.

---

## Project
**Name:** Sterling — Jarvis-inspired local AI room assistant  
**Owner:** jtb  
**CWD:** `/Users/jtb/src/sterling`  
**Venv:** `ster/` (activate: `source ster/bin/activate`)  
**Status:** M1 Mac v1 — fully operational, all integrations live

---

## Stack (M1 Mac v1)
| Layer | Tech |
|---|---|
| Wake Word | Faster-Whisper `tiny` + energy VAD — any phrase, no training |
| STT | Faster-Whisper `base`, CPU, int8 |
| LLM | Ollama + `llama3.2:3b` via `http://localhost:11434` |
| TTS | Edge-TTS `en-GB-RyanNeural` + macOS `say` fallback |
| Lights | Govee cloud HTTP API |
| Music | Spotify Web API via spotipy (Premium required) |
| Weather | wttr.in — free, no key |
| Vision | USB webcam + YOLOv8n + face_recognition + object tracker + gesture detector |
| Memory | JSON archive (ChromaDB broken on Python 3.14 — falls back gracefully) |
| Audio | Single shared PyAudio stream (wake word + recorder share one CoreAudio input) |
| Config | `config.yaml` (gitignored — has API keys) |

---

## File Map
```
main.py                      ← orchestrator; CLI: --no-vision, --config
core/wake_word.py            ← Faster-Whisper wake word + energy VAD
core/stt.py                  ← Faster-Whisper STT
core/llm.py                  ← Ollama client (chat + stream)
core/tts.py                  ← Edge-TTS sentence-streaming + stop_event interruption
core/memory.py               ← session window + JSON archive (ChromaDB optional)
core/govee.py                ← GoveeCloud HTTP client
core/spotify.py              ← Spotify Web API via spotipy
core/workspace.py            ← project scaffolding + LLM code gen
vision/__init__.py           ← vision package
vision/webcam.py             ← WebcamVision — YOLO + face_recognition + scene description
                                + enroll_face() + check_presence() + get_scene_description()
vision/object_tracker.py     ← JSON-backed last-seen tracker for everyday objects
vision/gesture.py            ← YOLOv8n-pose background gesture detector
vision/faces/                ← drop named JPGs here to enrol faces
utils/audio.py               ← AudioRecorder — shared persistent stream mode
utils/weather.py             ← wttr.in fetch + two-day forecast parser
utils/text.py                ← markdown stripper for TTS
utils/logger.py              ← shared logger
scripts/discover_govee.py    ← Govee cloud discovery helper
prompts/system_prompt.txt    ← Sterling personality
config.yaml                  ← runtime config (gitignored)
config.yaml.example          ← safe-to-commit template (secrets redacted)
memory.json                  ← JSON message archive (gitignored)
object_tracker.json          ← last-seen object positions (gitignored)
FUTURE_ITERATIONS.md         ← full roadmap: platforms + 30+ feature ideas
README.md                    ← quick-start
```

---

## Key Decisions & Architecture

### Audio — Single Shared Stream (CRITICAL)
- **One PyAudio stream for the entire session.** Wake word detector and recorder share it.
- `recorder.enable_shared_mode()` opens the stream once; `close_stream()` / `open_stream()` are no-ops.
- `wake_word.start(external_read=recorder.read_raw)` — detector reads via callback, no own stream.
- `wake_word.pause()` / `resume()` = VAD state reset only (no stream ops).
- This eliminates `||PaMacCore (AUHAL)|| err=-50` (two simultaneous CoreAudio inputs).
- **Previous broken approach:** closing recorder → opening wake word stream was a race that segfaulted.

### `_speak_interruptible()` ordering (CRITICAL — segfault if wrong)
```
stop_event.set()
monitor.join(timeout=0.5)   ← wait for monitor to exit (~32ms with stop_event in listen())
wake_word.pause()           ← VAD reset (no-op on stream since shared mode)
recorder.open_stream()      ← no-op in shared mode
```
- Monitor passes `stop_event` to `listen()` → skips Whisper transcription → exits in ~32ms.
- DO NOT open recorder before monitor exits — was the original segfault cause.

### Ctrl+C
- Signal handler: save memory → `os._exit(0)`. Force exit bypasses `Pa_StopStream` deadlock on macOS.

### Conversation Loop
- `_handle_command()` returns True for ALL built-in commands.
- `_conversation_loop` only `break`s when `not self._running` (real shutdown). Everything else `continue`s.
- Vision, time, weather, tracking → context injected into LLM prompt (same pattern as weather).

### Memory Recall
- JSON recall (`recall_turns: 2`) wraps old turns in a system message labelled "PREVIOUS session".
- **Do NOT inject recalled turns as bare user/assistant messages** — LLM treats them as current topic.
- ChromaDB broken on Python 3.14 (`chromadb_rust_bindings` missing) — graceful JSON fallback.

### Vision Pipeline
- `get_scene_description()` — single YOLO pass; face recognition + expression + activity + tracker update.
- Expression: smile detection via dlib lip landmarks (`smile_threshold` config).
- Activity: object-combo inference (laptop+mouse = "working at computer", phone near face = "on phone").
- Object tracker updated as side-effect of every scene query — no background YOLO loop.
- `enroll_face(name)` — voice-triggered snapshot, saves to `vision/faces/`, reloads immediately.
- `check_presence()` — lightweight bool for presence monitor.

### Gesture Detection (`vision/gesture.py`)
- YOLOv8n-pose, daemon thread, 1Hz, 1.5s sustain before firing.
- Gestures: `wave` (wrist above nose) → wake, `hands_up` → stop audio, `point_right/left` → Spotify.
- **Disabled by default** (`gestures.enabled: false`). Pose model auto-downloads on first use (~7MB).
- Thumbs up/down needs MediaPipe Hands (finger keypoints) — tracked in FUTURE_ITERATIONS.md.

### Object Tracking (`vision/object_tracker.py`)
- Persists last-seen position + timestamp of ~20 everyday objects to `object_tracker.json`.
- Updated every time `get_scene_description()` is called — zero background cost.
- Queried by "where's my phone?" / "have you seen my keys?" → injected as LLM context.

### Wake Word
- `"ling"` removed from phrases — was matching substrings ("darling", "Sterling" mid-word).
- Current phrases: `["hey sterling", "sterling", "ster"]`.

### Config
- **Everything is configurable.** Phrase lists, response strings, thresholds — all in `config.yaml`.
- `commands.*` — trigger phrases for shutdown, winddown, memory_clear, session_status, diagnostics.
- `queries.*` — trigger phrases for vision, time, weather, tracking context injectors.
- `sterling.*` — all spoken responses (ack_message, sleep_message, etc.).
- `tts.stream_chunk_chars` — streaming synthesis chunk size.
- `vision.face_tolerance`, `vision.face_merge_distance`, `vision.smile_threshold`.
- `gestures.point_min_distance`, `gestures.point_max_height`.
- `_phrases(section, key, defaults)` helper in main.py — reads config, falls back to hardcoded defaults.

### TTS
- `stream_chunk_chars=80` — min chars before synthesis chunk fires. Lower = first words faster.
- `speak_streaming()` join polls at 50ms (was 200ms) for tighter latency.

### Startup Warnings (Cannot Fix)
- `objc[]: Class AVFFrameReceiver implemented in both av/libavdevice.62 and cv2/libavdevice.61` — C-level dylib conflict, fires before Python, harmless in practice.
- ChromaDB `WARNING: init failed` — expected on Python 3.14, JSON recall handles it.

---

## Hardware (Owner)
- **Current:** M1 Mac 8GB — dev/daily driver
- **Planned:** Jetson Orin Nano (owned)
  - TTS: Edge-TTS → Piper (offline)
  - systemd service for boot-time auto-start
  - YOLO → GPU inference (much faster), larger models viable

---

## Needs / Next Up
- [ ] Enrol jtb's face in `vision/faces/jtb.jpg` for face recognition
- [ ] Tune `silence_threshold` and `wake_word.energy_threshold` for mic/room acoustics
- [ ] Enable gestures (`gestures.enabled: true`) and test wave-to-wake
- [ ] Enable presence detection (`presence.enabled: true`) and tune `check_interval`
- [ ] Check Jetson RAM: `free -h` — 4GB vs 8GB determines LLM model
- [ ] Jetson migration: JetPack 6, Ollama ARM, Piper TTS, systemd service
- [ ] Implement timers/reminders (top of backlog — high daily value)
- [ ] Implement morning briefing (second priority — peak Jarvis moment)
- [ ] Implement clipboard assistant (`pip install pyperclip` — trivial, huge dev QoL)
- [ ] VLM upgrade: `ollama pull moondream` → replace YOLO text description with actual pixel vision

---

## Update Rules
- Update when: new files added, stack changes, big decisions made, phase completes
- Do NOT update for: minor edits, bug fixes, small tweaks
- Keep it scannable — bullet points over paragraphs
