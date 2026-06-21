# Sterling — AI Memory File
> Read this at the start of every session. Update only on meaningful changes (new files, decisions, direction shifts). Keep it tight.

---

## Project
**Name:** Sterling — Jarvis-inspired local AI room assistant  
**Owner:** jtb  
**CWD (Mac dev):** `/Users/jtb/src/sterling`  
**CWD (Windows):** Windows folder is the active build target  
**Venv:** `ster/` (Mac: `source ster/bin/activate` | Windows: `ster\Scripts\activate`)  
**Status:** M1 Mac v1 fully operational. Windows build — code ported, setup guide written, ready to deploy.
**Last session:** 2026-06-20 — Windows port: TTS afplay→pygame, say→pyttsx3, config updated for RTX 3060 (CUDA, large-v3 STT, qwen2.5:7b LLM), WINDOWS_SETUP.md created, setup_windows.bat created

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

## Stack (Windows v2 — RTX 3060 12 GB / 16 GB RAM)
| Layer | Tech |
|---|---|
| Wake Word | Faster-Whisper `tiny` + energy VAD, CPU, int8 (unchanged) |
| STT | Faster-Whisper `large-v3`, **CUDA**, float16 |
| LLM | Ollama + `qwen2.5:7b` via `http://localhost:11434` (CUDA auto) |
| TTS | Edge-TTS `en-GB-RyanNeural` + **pygame.mixer** playback + **pyttsx3** fallback |
| Lights | Govee cloud HTTP API (unchanged) |
| Music | Spotify Web API via spotipy (unchanged) |
| Weather | wttr.in (unchanged) |
| Vision | USB webcam + YOLOv8n/s + face_recognition + object tracker + gesture detector |
| Memory | JSON archive + keyword recall (ChromaDB optional on Python 3.11) |
| Audio | Single shared PyAudio stream (WASAPI backend on Windows) |
| Config | `config.yaml` — Windows version with CUDA settings |

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

### Memory — REWRITTEN 2026-06-14 (core/memory.py)
- **Three layers:** (1) clean session window, (2) `KeywordRecall` (pure-Python IDF-weighted
  retrieval over memory.json — replaces broken ChromaDB), (3) `FactStore` (facts.json, durable).
- **CRITICAL — context pollution fix:** the session window now stores ONLY the clean user
  utterance. Per-turn system data (weather/vision/Spotify action/time) is injected via
  `get_messages(ephemeral_context=...)` for ONE generation and never stored. This fixes the
  "play music then ask for a joke → talks about music" bug. Verified with llama3.2:3b.
- **Do NOT** go back to baking brackets into the stored user text (`add_user(llm_text)`). Store
  `text`, pass brackets as `ephemeral`.
- `KeywordRecall.query()` scores past exchanges by IDF-weighted keyword overlap; only injects when
  relevant (irrelevant queries return nothing). Works on Python 3.14, no native deps.
- `FactStore`: "remember that I like jazz" → stored to facts.json, injected by relevance + recency.
  Voice cmd "forget what you know about me" wipes it. `_parse_remember_fact()` excludes face-enroll phrasing.
- ChromaDB now `chroma_enabled: false` by default (rust bindings missing on 3.14). Keyword recall is the engine.
- `Memory.pop_last_user()` still removes orphaned user turn on TTS interrupt.

### Self-Echo Interrupt Suppression (NEW — fixes self-interruption)
- Root cause: interrupt monitor runs wake-word detection through the mic during TTS; with no AEC,
  Sterling hears himself and the short wake phrases echo-match.
- Fix: `TTS.spoken_text` exposes what's being said right now (updated token-by-token in the streaming
  producer). `_speak_interruptible` calls `wake_word.enable_echo_guard(lambda: tts.spoken_text)`.
- `WakeWordDetector._is_self_echo()` rejects a transcript if ≥50% of its words appear in Sterling's
  current speech. "sterling" (which Sterling never says, per system prompt) won't be in spoken text,
  so genuine barge-in still works. Guard disabled again in the `finally` block.

### Vision Pipeline
- **VLM upgrade hook (NEW):** set `vision.vlm_model: "moondream"` (after `ollama pull moondream`) to
  get real pixel understanding for "what am I holding"/"describe the scene". `_describe_scene()` sends
  a JPEG frame (`webcam.get_frame_jpeg()`) to `LLM.describe_image()` and pairs the VLM answer with
  YOLO labels for grounding. Blank `vlm_model` → fast YOLO+heuristic path (unchanged). moondream ~1.7GB, fits 8GB M1.
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
- `_speak_interruptible` monitor has a **1.5 s grace period** (`stop_event.wait(timeout=1.5)`) before it starts
  accepting wake word detections. Prevents Sterling's own voice ("Yes?", name said mid-response) from
  echo-triggering through the mic. If TTS finishes in < 1.5 s, `stop_event` fires and monitor exits cleanly.
- System prompt now instructs Sterling to **never say its own name** — removes the mid-response self-trigger root cause.

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

### Spotify Intent Parsing
- `_parse_spotify_intent` strips trailing filler from play queries: `" for me"`, `" please"`, `" right now"`, `" now"`, `" on spotify"`. Prevents `"chance the rapper for me?"` from being searched literally.
- After `_execute_spotify_command`, `spotify_context` is now set for **all** action types (play, pause, skip, previous, volume), not just `now_playing`. Context is injected into `llm_text` as `[Action completed: ...]` so LLM knows what was done and responds naturally.

### Startup Warnings (Cannot Fix)
- `objc[]: Class AVFFrameReceiver implemented in both av/libavdevice.62 and cv2/libavdevice.61` — C-level dylib conflict, fires before Python, harmless in practice.
- ChromaDB `WARNING: init failed` — expected on Python 3.14, JSON recall handles it.

---

## Hardware (Owner)
- **Mac dev machine:** M1 Mac 8GB — original dev/daily driver
- **Windows target:** RTX 3060 12 GB / 16 GB RAM — Windows build live
  - TTS: `afplay`→`pygame.mixer`, `say`→`pyttsx3`
  - STT: large-v3 on CUDA (was base on CPU)
  - LLM: qwen2.5:7b (was llama3.2:3b)
- **Planned:** Jetson Orin Nano (owned)
  - TTS: Edge-TTS → Piper (offline)
  - systemd service for boot-time auto-start
  - YOLO → GPU inference (much faster), larger models viable

---

## Needs / Next Up
- [ ] **Windows: Deploy and boot-test** — install Python 3.11, CUDA, Ollama, run setup_windows.bat
- [ ] **Windows: PyAudio install** — may need pipwin or pre-built wheel on Windows
- [ ] **Windows: Verify pygame audio playback** — test speak() and speak_streaming()
- [ ] **Windows: Verify CUDA** — `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Enrol face in `vision/faces/` for face recognition
- [ ] Tune `silence_threshold` and `wake_word.energy_threshold` for mic/room acoustics
- [ ] Enable gestures (`gestures.enabled: true`) and test wave-to-wake
- [ ] Enable presence detection (`presence.enabled: true`) and tune `check_interval`
- [ ] Wire in Kokoro-82M offline TTS (hooks ready, not yet integrated)
- [ ] Enable ChromaDB semantic memory on Python 3.11 (`chroma_enabled: true`)
- [ ] Pull moondream VLM: `ollama pull moondream` → set `vlm_model: "moondream"` in config
- [ ] Try `qwen2.5:14b` if 7b feels too basic (8.7 GB VRAM — tight with vision, fine without)
- [ ] Implement timers/reminders (top of backlog — high daily value)
- [ ] Implement morning briefing (second priority — peak Jarvis moment)
- [ ] Implement clipboard assistant (`pip install pyperclip` — trivial, huge dev QoL)
- [ ] Jetson migration: JetPack 6, Ollama ARM, Piper TTS, systemd service

---

## Update Rules
- Update when: new files added, stack changes, big decisions made, phase completes
- Do NOT update for: minor edits, bug fixes, small tweaks
- Keep it scannable — bullet points over paragraphs
