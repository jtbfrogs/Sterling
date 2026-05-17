# Sterling — AI Room Assistant

> *"Sterling online. All systems nominal. How may I assist you?"*

A local, voice-first AI assistant for your workspace. Jarvis-inspired. No cloud required.

---

## Quick Start (M1 Mac)

```bash
# 1. Run setup (installs everything)
bash setup_mac.sh

# 2. Get a free Picovoice key → https://picovoice.ai
#    Paste it into config.yaml under porcupine.access_key

# 3. Launch
source ster/bin/activate
python main.py
```

Say **"Jarvis"** → wait for *"Yes?"* → speak your request.

---

## Documentation

| Document | Description |
|---|---|
| [`STERLING.md`](STERLING.md) | Full system breakdown, architecture, components, setup, configuration |
| [`FUTURE_ITERATIONS.md`](FUTURE_ITERATIONS.md) | Windows (GTX 3060) and Linux build plans |

---

## What Sterling Can Do

- **Help with projects** — planning, architecture, debugging, code review
- **Brainstorm ideas** — concepts, features, names, solutions
- **Answer questions** — broad knowledge across all domains
- **Room awareness** — face recognition, object detection (HuskyLens2)
- **Hold a conversation** — maintains context across the session

---

## Voice Commands

| Say | Action |
|---|---|
| `"Jarvis"` | Wake word — activates Sterling |
| `"Clear memory"` | Reset conversation history |
| `"Who's in the room?"` | Report detected faces (vision required) |
| `"What do you see?"` | Report detected objects (vision required) |
| `"How long have we been talking?"` | Session duration |
| `"Goodbye"` / `"Shut down"` | Graceful shutdown |

---

## Stack

| Component | Technology |
|---|---|
| Wake Word | Porcupine (Picovoice) |
| Speech-to-Text | Faster-Whisper (base model) |
| Language Model | Ollama + Llama 3.2 3B |
| Text-to-Speech | Edge-TTS (en-GB-RyanNeural) |
| Vision | HuskyLens2 via USB-Serial |
| Platform | M1 Mac, Python 3.11, Metal GPU |

---

## Options

```bash
python main.py --no-vision          # Skip HuskyLens2 initialization
python main.py --config dev.yaml    # Use alternate config file
```

---

*Read [`STERLING.md`](STERLING.md) for the full breakdown.*
