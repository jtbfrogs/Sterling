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
python main.py --no-vision    # if HuskyLens2 isn't connected
```

Say **"Hey Sterling"** → wait for *"Yes?"* → speak your request.

---

## What Sterling Can Do

- **Conversation** — project help, debugging, brainstorming, general knowledge
- **Smart wind-down** — say *"goodbye"*, *"I'm done"*, or *"talk later"* and Sterling says goodbye then goes back to listening
- **Room awareness** — face recognition and object detection via HuskyLens2
- **Control Govee lights** — on/off, colors, brightness, mid-conversation
- **Spotify control** — play, pause, skip, volume, search by artist or song
- **Weather** — current conditions and two-day forecast, any location
- **Persistent memory** — remembers context across sessions via `memory.json`

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
| `"Who's in the room?"` | Report detected faces (vision required) |
| `"What do you see?"` | Report detected objects (vision required) |
| `"Clear memory"` | Reset conversation history |
| `"How long have we been talking?"` | Session duration |
| `"I'm done"` / `"Goodbye"` / `"Talk later"` | Sterling says goodbye, returns to wake word |
| `"Shut down"` / `"Power off"` | Full shutdown |

---

## Stack

| Component | Technology |
|---|---|
| Wake Word | Faster-Whisper tiny + energy VAD |
| Speech-to-Text | Faster-Whisper base |
| Language Model | Ollama + Llama 3.2 1B |
| Text-to-Speech | Edge-TTS (`en-GB-RyanNeural`) |
| Smart Lights | Govee cloud API |
| Music | Spotify Web API (spotipy) |
| Weather | wttr.in (no API key) |
| Vision | HuskyLens2 via USB-Serial |
| Memory | JSON (session + persistent cross-session recall) |
| Platform | M1 Mac, Python 3.11, Metal GPU |

---

## Setup — Integrations

### Govee Lights
```bash
python scripts/discover_govee.py   # finds your devices and prints config
```
Then set `govee.enabled: true` and paste the device block into `config.yaml`.

### Spotify
1. Create a free app at [developer.spotify.com](https://developer.spotify.com) — tick **Web API**
2. Add redirect URI: `http://localhost:8888/callback`
3. Paste `client_id` and `client_secret` into `config.yaml`
4. Set `spotify.enabled: true` — first launch opens a browser for a one-time login

### Weather
Set your default location in `config.yaml` under `weather.location`. No API key needed.
You can also ask for any city mid-conversation: *"what's the weather in Tokyo?"*

---

## Configuration

Copy the example and fill in your details:
```bash
cp config.yaml.example config.yaml
```

Key sections in `config.yaml`:

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
| `vision` | HuskyLens2 port and face map |
| `memory` | History size, persistence, cross-session recall |
| `conversation` | Idle timeout, sleep message |

---

## CLI Options

```bash
python main.py                      # Default
python main.py --no-vision          # Skip HuskyLens2 initialisation
python main.py --config dev.yaml    # Use alternate config file
```

---

## Documentation

| Document | What's in it |
|---|---|
| [`STERLING.md`](STERLING.md) | Full architecture, components, setup, configuration reference |
| [`FUTURE_ITERATIONS.md`](FUTURE_ITERATIONS.md) | Jetson Orin build, Windows GPU build, ideas & roadmap |
| [`VOICES.md`](VOICES.md) | Full Edge-TTS and macOS voice reference |

---

*Platform: M1 Mac v1 — Jetson Orin build planned. Read [`FUTURE_ITERATIONS.md`](FUTURE_ITERATIONS.md) for the roadmap.*
