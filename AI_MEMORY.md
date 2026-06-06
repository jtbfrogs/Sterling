# Sterling — AI Memory File
> Read this at the start of every session. Update only on meaningful changes (new files, decisions, direction shifts). Keep it tight.

---

## Project
**Name:** Sterling — Jarvis-inspired local AI room assistant  
**Owner:** jtb  
**CWD:** `/Users/jtb/src/sterling`  
**Venv:** `ster/` (activate: `source ster/bin/activate`)  
**Status:** M1 Mac v1 — operational, all integrations live

---

## Stack (M1 Mac v1)
| Layer | Tech |
|---|---|
| Wake Word | Faster-Whisper `tiny` + energy VAD — any phrase, no training, no API key |
| STT | Faster-Whisper `base`, CPU, int8 |
| LLM | Ollama + `llama3.2:3b` via `http://localhost:11434` |
| TTS | Edge-TTS `en-GB-RyanNeural` + macOS `say` fallback |
| Lights | Govee cloud HTTP API (api_key + device_id + model SKU) |
| Music | Spotify Web API via spotipy (Premium required) |
| Weather | wttr.in — free, no key, any location on the fly |
| Vision | USB webcam + YOLOv8 + face_recognition |
| Memory | ChromaDB semantic recall + JSON archive |
| Audio | PyAudio + energy-based VAD |
| Config | `config.yaml` (gitignored — has API keys) |

---

## File Map
```
main.py                    ← orchestrator; CLI: --no-vision, --config
core/wake_word.py          ← Faster-Whisper wake word + energy VAD
core/stt.py                ← Faster-Whisper STT
core/llm.py                ← Ollama client (chat + stream)
core/tts.py                ← Edge-TTS sentence-streaming + stop_event interruption
core/memory.py             ← ChromaMemory (semantic) + session window + JSON archive
core/govee.py              ← GoveeCloud HTTP client
core/spotify.py            ← Spotify Web API via spotipy
core/workspace.py          ← project scaffolding (Python/C/C++/JS/Rust/Go/Bash) + LLM code gen
vision/__init__.py         ← vision package
vision/webcam.py           ← WebcamVision — YOLOv8 detection + face_recognition enrollment
vision/faces/              ← drop named JPGs here to enrol faces
utils/audio.py             ← VAD recorder
utils/weather.py           ← wttr.in fetch + two-day forecast parser
utils/text.py              ← markdown stripper for TTS
utils/logger.py            ← shared logger
scripts/discover_govee.py  ← cloud API discovery helper (prints config to paste)
prompts/system_prompt.txt  ← Sterling personality + system status injected at runtime
config.yaml                ← runtime config (gitignored)
config.yaml.example        ← safe-to-commit template
memory.json                ← JSON message archive (gitignored)
.chroma/                   ← ChromaDB vector store (gitignored)
requirements_mac.txt       ← pip deps
STERLING.md                ← full system doc
FUTURE_ITERATIONS.md       ← Jetson Orin + Windows build + ideas roadmap
README.md                  ← quick-start, voice commands, setup
examples/                  ← standalone reference implementations
```

---

## Key Decisions Made
- Wake word: Faster-Whisper `tiny` + energy VAD — no training, no API key
- LLM: `llama3.2:3b`, max_tokens 150, short/casual system prompt — fixes verbose responses
- Govee: cloud HTTP API — `GoveeCloud` class, devices: Bed lights (H61BE) + TV Backlight (H6099)
  - Intent parsed pre-LLM; action fires immediately; Ollama always speaks the response
- Spotify: spotipy Web API; Premium required; OAuth token cached at `.spotify_cache`
  - Intent parser extracts search query from natural speech; "what's playing" injects track to LLM
- Weather: wttr.in — current + two-day forecast with precip % and high/low temps
  - Location extracted from speech on the fly; falls back to config default
- Conversation wind-down: "i'm done" / "goodbye" / "talk later" → LLM says goodbye → wake word
  - Hard shutdown (kills process): "shut down" / "power off" / "go offline"
- Wake word interruption: wake phrase mid-response stops TTS immediately
  - `_speak_interruptible()` swaps mic: recorder → wake word detector during TTS
  - Full Whisper phrase match (not energy) — stops on wake word only
  - stop_event → afplay killed within 50ms; partial response discarded from memory
- Intent mutual exclusion: Spotify, lights, project creation checked in order with `intent_handled` flag
- Vision: USB webcam + YOLOv8 + face_recognition — HuskyLens fully removed
  - `vision/webcam.py` — background capture thread, YOLO object detection, face ID from `vision/faces/`
  - System status injected into prompt at startup so LLM knows what's online/offline
  - Vision queries intercepted before LLM if camera offline — honest "camera offline" response
- ChromaDB semantic memory: every exchange embedded + stored in `.chroma/`
  - On each LLM call: semantic query injects top 3 relevant past exchanges with timestamps
  - Session window: 10 turns | JSON recency anchor: 2 turns
  - Embedding: all-MiniLM-L6-v2 ONNX, one-time 79MB download, auto-cached
- Workspace: project creation at `/Users/jtb/src/sterling`
  - Python (+venv), C, C++, JavaScript, Rust, Go, Bash
  - Separate 1024-token code-only LLM prompt for generation
- ChromaDB context injection explicitly timestamped and labelled as past — prevents LLM treating old exchanges as current
- Diagnostics command: "run diagnostics" / "system status" → speaks what's online/offline

---

## Hardware (Owner)
- **Current:** M1 Mac 8GB — dev/daily driver
- **Planned:** Jetson Orin Nano (owned — check `free -h` for 4GB vs 8GB)
  - Dedicated always-on Sterling box
  - TTS swap: Edge-TTS → Piper (offline) on Jetson
  - systemd service for boot-time auto-start

---

## Needs / Next Up
- [ ] Test webcam vision end-to-end (`pip install ultralytics dlib opencv-python face_recognition`)
- [ ] Tune `silence_threshold` and `wake_word.energy_threshold` for jtb's mic/room
- [ ] Verify wake word interruption feels natural (~0.8–1s stop delay expected)
- [ ] Check Jetson RAM: `free -h` — 4GB or 8GB determines LLM model choice
- [ ] Jetson migration: JetPack 6, Ollama ARM, Piper TTS, systemd service
- [ ] Consider removing `"ling"` from wake phrases if false triggers happen

---

## Update Rules
- Update when: new files added, stack changes, big decisions made, phase completes
- Do NOT update for: minor edits, bug fixes, small tweaks
- Keep it scannable — bullet points over paragraphs
