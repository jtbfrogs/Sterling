# Sterling — AI Memory File
> Read this at the start of every session. Update only on meaningful changes (new files, decisions, direction shifts). Keep it tight.

---

## Project
**Name:** Sterling — Jarvis-inspired local AI room assistant  
**Owner:** jtb  
**CWD:** `/Users/jtb/src/sterling`  
**Venv:** `ster/` (activate: `source ster/bin/activate`)  
**Status:** M1 Mac v1 — core built, not yet tested end-to-end

---

## Stack (M1 Mac v1)
| Layer | Tech |
|---|---|
| Wake Word | faster-whisper `tiny` + energy VAD — any phrase, no training, no API key |
| STT | Faster-Whisper `base`, CPU, int8 |
| LLM | Ollama + `llama3.2` via `http://localhost:11434` |
| TTS | Edge-TTS `en-GB-RyanNeural` + macOS `say` fallback |
| Lights | Govee Local LAN API (UDP, no cloud) |
| Vision | HuskyLens2 USB-Serial (UART 9600 baud) |
| Audio | PyAudio + energy-based VAD |
| Config | `config.yaml` (never commit — has API key) |

---

## File Map
```
main.py                           ← orchestrator, CLI flags: --no-vision, --config
core/wake_word.py                 ← openWakeWord wrapper (custom + pre-trained)
scripts/train_wake_word.py        ← custom ONNX wake word trainer (no PyTorch)
assets/hey_sterling.onnx          ← trained wake word model (generated, gitignored)
core/stt.py              ← Faster-Whisper
core/llm.py              ← Ollama client (chat + stream)
core/tts.py              ← Edge-TTS, sentence-streaming, fallback
core/memory.py           ← sliding window context (session only)
core/vision.py           ← HuskyLens2 binary UART protocol
utils/audio.py           ← VAD recorder
utils/text.py            ← markdown stripper for TTS
utils/logger.py          ← shared logger
core/govee.py            ← Govee Local LAN client (UDP control + discovery)
scripts/discover_govee.py ← standalone script to find Govee device IPs
prompts/system_prompt.txt ← Sterling personality (no markdown in responses)
config.yaml              ← runtime config (gitignored)
config.yaml.example      ← safe-to-commit template
VOICES.md                ← full Edge-TTS + macOS say voice reference list
requirements_mac.txt     ← pip deps
setup_mac.sh             ← one-command setup script
STERLING.md              ← full system doc
FUTURE_ITERATIONS.md     ← Windows (3060) + Linux plans
```

---

## Key Decisions Made
- Wake word uses faster-whisper `tiny` model + energy VAD (no openWakeWord at runtime)
  - Any phrase works immediately — no training required
  - Config: `wake_word.phrases` list (default: hey sterling, sterling, ster, ling)
  - State machine: WAITING → RECORDING → transcribe → phrase match → trigger
  - `scripts/train_wake_word.py` still available for optional openWakeWord custom model
- Logging duplicate bug fixed: `logger.propagate = False` in `utils/logger.py`
- Vision: `ping()` fixed (was sending `0x37`, now correct `0x2C` knock command); startup_scan() added
- TTS: rewrote with synthesis/playback pipeline (two threads + queue) — no more gaps between sentences
- TTS: ellipsis `...` no longer splits sentences (was causing many tiny HTTP calls)
- TTS: `stop()` method + `stop_event` param added for interruption support
- Interruption monitor removed — was picking up Sterling's own TTS output and cutting itself off
- Conversation mode: wake once, talk freely — `_conversation_loop()` via `listen_for_speech(timeout)`
- One stream at a time rule: wake word stream paused for entire interaction, resumed after
- TTS sentence-streams: Sterling speaks sentence 1 while LLM generates the rest
- System prompt forbids markdown output (TTS reads it aloud otherwise)
- Govee lights integrated via Local LAN UDP API — no cloud, no API key needed
  - Light intent parsed in main.py before LLM call; action fires immediately
  - LLM always speaks the response (not hardcoded strings) — stays in conversation flow
  - Enable in config: govee.enabled: true + govee.devices list with IPs
  - Get IPs: python scripts/discover_govee.py
  - Supported commands: on/off, color by name (15 colors), brightness (% or dim/bright keywords)
- LLM `max_tokens` (Ollama `num_predict`) wired through config.yaml → main.py → llm.py
  - Default: 512 | Brief: 128–256 | Detailed: 1024+
- Vision is optional — `--no-vision` flag or `vision.enabled: false`
- Memory now persists to `memory.json` — every message timestamped, sessions tracked with start/end times
- On startup, last `recall_turns` (default 10) turns from previous sessions injected into context window
- Atomic save (write .tmp → rename) prevents corruption on crash; `end_session()` called in `shutdown()`
- `memory.json` gitignored; ChromaDB still planned for v2 (semantic search)
- `config.yaml` is gitignored; `config.yaml.example` is the safe template

---

## Needs / Next Up
- [ ] End-to-end test on real hardware
- [ ] Test Govee integration on real hardware (config device IPs first)
- [ ] Tune `silence_threshold` for jtb's mic
- [ ] Tune `wake_word.energy_threshold` and `audio.silence_threshold` for jtb's mic/room
- [ ] Tune `conversation.timeout` (currently 20s) for preferred idle time
- [ ] Consider removing `"ling"` from phrases if false triggers happen
- [ ] Verify HuskyLens2 ping works (protocol fix deployed)
- [ ] Optionally bump HuskyLens baud rate to 115200 on device + config
- [x] JSON persistent memory (done — timestamped, session-aware, auto-recalled on startup)
- [ ] ChromaDB long-term memory (semantic search — v2)
- [ ] Interruption handling (needs speaker/mic separation or VAD gating before re-attempting)
- [ ] Windows v2 build (when ready)

---

## Update Rules
- Update this file when: new files added, stack changes, big decisions made, phase completes
- Do NOT update for: minor edits, bug fixes, small tweaks
- Keep total length under ~60 lines
