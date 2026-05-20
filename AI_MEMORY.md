# Sterling — AI Memory File
> Read this at the start of every session. Update only on meaningful changes (new files, decisions, direction shifts). Keep it tight.

---

## Project
**Name:** Sterling — Jarvis-inspired local AI room assistant  
**Owner:** jtb  
**CWD:** `/Users/jtb/src/sterling`  
**Venv:** `ster/` (activate: `source ster/bin/activate`)  
**Status:** M1 Mac v1 — core operational, integrations live

---

## Stack (M1 Mac v1)
| Layer | Tech |
|---|---|
| Wake Word | Faster-Whisper `tiny` + energy VAD — any phrase, no training, no API key |
| STT | Faster-Whisper `base`, CPU, int8 |
| LLM | Ollama + `llama3.2:3b` via `http://localhost:11434` |
| TTS | Edge-TTS `en-GB-RyanNeural` + macOS `say` fallback |
| Lights | Govee cloud HTTP API (api_key + device_id + model/SKU) |
| Music | Spotify Web API via spotipy (Premium required for playback control) |
| Weather | wttr.in — no API key, free, supports any location on the fly |
| Vision | HuskyLens2 USB-Serial (UART 9600 baud) |
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
core/memory.py             ← session window + ChromaDB semantic recall + JSON archive
core/workspace.py          ← project scaffolding (Python/C/C++/JS/Rust/Go/Bash) + LLM code gen
core/vision.py             ← HuskyLens2 binary UART protocol
core/govee.py              ← GoveeCloud (HTTP) + GoveeLocal (UDP) — same interface
core/spotify.py            ← Spotify Web API via spotipy (play/pause/skip/volume/search)
utils/audio.py             ← VAD recorder
utils/weather.py           ← wttr.in fetch + two-day forecast parser
utils/text.py              ← markdown stripper for TTS
utils/logger.py            ← shared logger
scripts/discover_govee.py  ← cloud API + local LAN discovery helper
prompts/system_prompt.txt  ← Sterling personality (short, casual, no markdown)
config.yaml                ← runtime config (gitignored)
config.yaml.example        ← safe-to-commit template
memory.json                ← persistent conversation memory (gitignored)
requirements_mac.txt       ← pip deps (includes spotipy)
STERLING.md                ← full system doc
FUTURE_ITERATIONS.md       ← Jetson Orin build + ideas roadmap
README.md                  ← up-to-date quick-start + voice command reference
```

---

## Key Decisions Made
- Wake word: Faster-Whisper `tiny` + energy VAD — no training, no Picovoice key needed
- LLM max_tokens dropped to 150 (config) + system prompt rewritten short/casual — fixes verbose responses
- Govee: uses cloud HTTP API (api_key + device_id + model SKU) — `GoveeCloud` class
  - `GoveeLocal` (UDP) also available if no api_key set — auto-selected in `_init_govee`
  - Both devices discovered live: Bed lights (H61BE) + TV Backlight 3 Lite (H6099)
  - Intent parsed pre-LLM; action fires immediately; Ollama always speaks response
- Spotify: spotipy + Web API; Premium required; OAuth token cached at `.spotify_cache`
  - Intent parser extracts search query from "play X" phrases
  - "what's playing" injects track info into LLM context
- Weather: wttr.in — current + two-day forecast with precip % and high/low
  - Location extracted from speech on the fly ("weather in Denver") — falls back to config default
  - Injected as directive context: "use only this data, do not guess"
- Conversation wind-down: "i'm done" / "goodbye" / "talk later" → LLM says goodbye → returns to wake word
  - "goodbye" removed from hard shutdown list (now wind-down only)
  - Hard shutdown: "shut down" / "power off" / "go offline"
- Wake word interruption: saying wake word mid-response stops TTS immediately
  - `_speak_interruptible()` swaps mic from recorder → wake word detector during TTS
  - Background thread calls `wake_word.listen()` (full Whisper phrase match — not just energy)
  - On detection: sets stop_event → afplay killed within 50ms
  - Partial response discarded from memory; conversation loop continues
  - Stream swap is the key — CoreAudio can't have two input streams simultaneously
- One stream at a time: wake word stream paused during conversation, recorder open throughout
- TTS sentence-streams: Sterling speaks sentence 1 while LLM generates the rest
- Model upgraded: llama3.2:1b → llama3.2:3b — smarter, still fits M1 8GB
- ChromaDB semantic long-term memory (replaces blind JSON recall injection)
  - Every exchange embedded + stored in `.chroma/` (gitignored)
  - On each LLM call: semantic query injects top 3 relevant past exchanges
  - Session window: 10 turns | JSON recency anchor: 2 turns
  - Embedding: all-MiniLM-L6-v2 ONNX, 79MB one-time download, auto-cached
  - Falls back to JSON recall if ChromaDB unavailable
- Workspace project creation at `/Users/jtb/src/sterling`
  - Python (+venv), C, C++, JavaScript, Rust, Go, Bash
  - Separate 1024-token code-only prompt for generation
- Memory persists to `memory.json` — timestamped, session-aware
- System prompt: short and casual — brevity rule is first instruction, no markdown

---

## Hardware (Owner)
- **Current:** M1 Mac 8GB — dev/daily driver
- **Planned:** Jetson Orin Nano (owned, version TBC — check `free -h` for RAM)
  - Jetson = dedicated always-on Sterling box
  - TTS swap: Edge-TTS → Piper (offline) when moving to Jetson
  - systemd service for boot-time auto-start

---

## Needs / Next Up
- [ ] Spotify: get client_id + client_secret from developer.spotify.com, enable in config
- [ ] Test Govee + Spotify + weather end-to-end on real hardware
- [ ] Tune `silence_threshold` and `wake_word.energy_threshold` for jtb's mic/room
- [ ] Verify wake word interruption timing feels natural (expect ~0.8–1s stop delay)
- [ ] Check Jetson Orin Nano RAM: `free -h` — 4GB or 8GB determines LLM model choice
- [ ] Jetson migration: flash JetPack 6, install Ollama ARM build, swap TTS to Piper
- [ ] Consider removing `"ling"` from wake phrases if false triggers happen
- [ ] Verify HuskyLens2 ping works (protocol fix deployed, untested on hardware)
- [ ] ChromaDB long-term memory (semantic search — v2)
- [ ] Voice notes feature (easy win — append to notes.md)
- [ ] Morning briefing command

---

## Update Rules
- Update when: new files added, stack changes, big decisions made, phase completes
- Do NOT update for: minor edits, bug fixes, small tweaks
- Keep it scannable — bullet points over paragraphs
