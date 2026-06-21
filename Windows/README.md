# Sterling — AI Room Assistant

> *"Sterling online. All systems nominal. How may I assist you?"*

A local, voice-first AI assistant for your workspace. Jarvis-inspired. Mostly offline.

---

## Quick Start (M1 Mac)

```bash
# 1. Activate the virtual environment
source ster/bin/activate

# 2. Make sure Ollama is running
ollama serve &

# 3. Launch
python main.py
python main.py --no-vision    # if camera isn't connected
```

Say **"Hey Sterling"** → wait for *"Yes?"* → speak your request.

---

## What Sterling Can Do

- **Conversation** — project help, debugging, brainstorming, general knowledge
- **Room awareness** — face recognition and object detection via USB webcam + YOLO
- **Control Govee lights** — on/off, colors, brightness, mid-conversation
- **Spotify control** — play, pause, skip, volume, search by artist or song
- **Weather** — current conditions and two-day forecast, any location
- **Create projects** — scaffold Python, C, C++, JS, Rust, Go, Bash with generated starter code
- **Persistent memory** — ChromaDB semantic recall across sessions
- **Smart wind-down** — say *"goodbye"*, *"I'm done"*, or *"talk later"* to end the conversation naturally
- **Wake word interrupt** — say the wake word mid-response to cut Sterling off immediately

---

## Voice Commands

| Say | Action |
|---|---|
| `"Hey Sterling"` / `"Sterling"` | Wake word — activates Sterling |
| `"Turn the lights off"` | Govee lights off |
| `"Make the lights blue"` | Govee color change (15 colors supported) |
| `"Dim the lights"` / `"Lights at 50%"` | Govee brightness |
| `"Play Radiohead"` / `"Play something"` | Spotify playback |
| `"Pause"` / `"Skip"` / `"Volume up"` | Spotify controls |
| `"What's playing?"` | Current Spotify track |
| `"What's the weather like?"` | Weather for your configured location |
| `"Weather in Denver"` | Weather for any city on the fly |
| `"Who's in the room?"` | Recognised faces from webcam |
| `"What do you see?"` | All detected objects from webcam |
| `"Create a Python project called tracker"` | Scaffold a new project |
| `"Run diagnostics"` / `"System status"` | What's online and what's not |
| `"Clear memory"` | Reset conversation history |
| `"I'm done"` / `"Goodbye"` / `"Talk later"` | Sterling says goodbye, returns to wake word |
| `"Shut down"` / `"Power off"` | Full shutdown |

---

## Stack

| Component | Technology |
|---|---|
| Wake Word | Faster-Whisper tiny + energy VAD |
| Speech-to-Text | Faster-Whisper base |
| Language Model | Ollama + Llama 3.2 3B |
| Text-to-Speech | Edge-TTS (`en-GB-RyanNeural`) |
| Smart Lights | Govee cloud API |
| Music | Spotify Web API (spotipy) |
| Weather | wttr.in (no API key) |
| Vision | USB webcam + YOLOv8 + face_recognition |
| Memory | ChromaDB semantic recall + JSON archive |
| Platform | M1 Mac, Python 3.11, Metal GPU |

---

## Setup — Integrations

### Vision (Webcam + YOLO)
```bash
source ster/bin/activate
brew install cmake                              # M1 Mac — needed for dlib
pip install ultralytics opencv-python
pip install dlib face_recognition               # optional — enables face ID
```

Set `vision.enabled: true` in `config.yaml`. To enrol faces drop a named photo into `vision/faces/`:
```
vision/faces/jtb.jpg    →  Sterling will say "I can see jtb"
```
Restart Sterling after adding photos.

---

### Govee Lights
```bash
python scripts/discover_govee.py   # finds your devices and prints config to paste
```
Then set `govee.enabled: true` and paste the device block into `config.yaml`.

---

### Spotify
1. Create a free app at [developer.spotify.com](https://developer.spotify.com) — tick **Web API**
2. Add redirect URI: `http://localhost:8888/callback`
3. Paste `client_id` and `client_secret` into `config.yaml`
4. Set `spotify.enabled: true` — first launch opens a browser for a one-time login

---

### Weather
Set your default location in `config.yaml` under `weather.location`. No API key needed.
You can also ask for any city on the fly: *"what's the weather in Tokyo?"*

---

## Configuration

Copy the example and fill in your details:
```bash
cp config.yaml.example config.yaml
```

| Section | Purpose |
|---|---|
| `sterling` | Name, startup/shutdown messages |
| `wake_word` | Trigger phrases, sensitivity |
| `llm` | Ollama model, temperature, max tokens |
| `stt` | Whisper model size, language |
| `tts` | Edge-TTS voice, rate, pitch |
| `govee` | API key and device list |
| `spotify` | Client credentials |
| `weather` | Default location |
| `workspace` | Where new projects get created |
| `vision` | Camera index, YOLO model, face enrollment directory |
| `memory` | Window size, ChromaDB path, persistence settings |
| `conversation` | Idle timeout, sleep message |

---

## CLI Options

```bash
python main.py                      # Default
python main.py --no-vision          # Skip vision initialisation
python main.py --config dev.yaml    # Use alternate config file
```

---

## Documentation

| Document | What's in it |
|---|---|
| [`STERLING.md`](STERLING.md) | Full architecture, components, setup, configuration reference |
| [`FUTURE_ITERATIONS.md`](FUTURE_ITERATIONS.md) | Jetson Orin build, Windows GPU build, ideas & roadmap |
| [`VOICES.md`](VOICES.md) | Full Edge-TTS voice reference |

---

*Platform: M1 Mac v1 — Jetson Orin Nano build planned. See [`FUTURE_ITERATIONS.md`](FUTURE_ITERATIONS.md).*
