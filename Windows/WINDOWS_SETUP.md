# Sterling — Windows Setup Guide
### RTX 3060 12 GB / 16 GB RAM Build

> A Jarvis-inspired local AI room assistant — voice-activated, runs 100% on your machine (no cloud AI required).

---

## Table of Contents
1. [What Sterling Does](#1-what-sterling-does)
2. [How It Works](#2-how-it-works)
3. [What You Need Before Starting](#3-what-you-need-before-starting)
4. [Step-by-Step Installation](#4-step-by-step-installation)
5. [Configuration](#5-configuration)
6. [Running Sterling](#6-running-sterling)
7. [Voice Commands](#7-voice-commands)
8. [Model Selection Guide](#8-model-selection-guide)
9. [RTX 3060 Performance Upgrades](#9-rtx-3060-performance-upgrades)
10. [Optional Features](#10-optional-features)
11. [Troubleshooting](#11-troubleshooting)
12. [How the Code Is Organized](#12-how-the-code-is-organized)

---

## 1. What Sterling Does

Sterling is a locally-running AI assistant you talk to out loud — no cloud AI, no subscription, no data leaving your machine.

**Core abilities:**
- 🎙️ **Wake word** — say "Sterling" or "Hey Sterling" and it wakes up
- 🗣️ **Voice conversation** — natural back-and-forth, remembers context
- 🎵 **Spotify control** — "play some Radiohead" / "skip this" / "turn it up"
- 💡 **Smart lights** — "turn the lights red" / "dim the bedroom lights"
- 🌤️ **Weather** — "what's the weather like?" uses live data
- 📷 **Vision** — webcam + YOLO to see the room, recognize faces, track objects
- 🧠 **Memory** — remembers things you tell it ("remember I like jazz")
- 🖥️ **Project scaffolding** — can generate code/projects via voice

**On your RTX 3060 specifically:**
- Whisper Large-v3 STT — near-perfect transcription accuracy
- 7B+ parameter LLM — much smarter than the 3B Mac default
- Real-time YOLO vision on GPU (60+ fps vs ~10 fps on CPU)
- Everything runs locally and offline (except Edge-TTS and Spotify)

---

## 2. How It Works

### The Main Loop

```
YOU SPEAK → [Wake Word Detector] → heard "Sterling"?
                                          ↓ yes
                               [Recorder] captures your speech
                                          ↓
                    [Whisper STT] converts audio → text
                                          ↓
               [LLM via Ollama] generates a response
                                          ↓
                      [Edge-TTS] speaks the response aloud
                                          ↓
                      back to waiting for wake word
```

### The Conversation Mode

After the wake word fires, Sterling stays "awake" for a multi-turn conversation — you don't have to say "Sterling" before every sentence. It goes back to sleep after 20 seconds of silence.

### Audio Architecture (Single Shared Stream)

Sterling uses **one microphone stream** for the entire session, shared between the wake word detector and the recorder. This prevents Windows audio conflicts (two apps fighting over the mic). During TTS playback, an interrupt monitor runs in a background thread — if you say something, it sets a stop signal and the audio cuts out within 50 ms.

### Self-Echo Guard

When Sterling is speaking, it knows what words it's saying. If the mic picks up its own voice through the speakers (no acoustic echo cancellation), the wake word detector compares the transcript to what Sterling is currently saying and ignores matches. A genuine wake word ("sterling" — which Sterling never says in its own responses) will still get through.

### Memory Layers

1. **Session window** — the last 12 turns of conversation (in RAM)
2. **Keyword recall** — archives all past sessions to `memory.json`, retrieved by IDF-weighted keyword matching (no ML, no native deps)
3. **Fact store** — durable facts you explicitly tell it ("remember I prefer dark mode")
4. **Optional ChromaDB** — semantic vector search over memory (requires Python 3.11, see Step 8 of install)

---

## 3. What You Need Before Starting

### Required Software

| Software | Why | Download |
|---|---|---|
| **Python 3.11** | Recommended version — best library compatibility | [python.org](https://www.python.org/downloads/) |
| **CUDA Toolkit 12.x** | GPU acceleration for Whisper and YOLO | [nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) |
| **Ollama for Windows** | Runs the LLM locally | [ollama.ai](https://ollama.ai) |
| **Microsoft C++ Build Tools** | Needed to compile dlib (face recognition) | [visualstudio.microsoft.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |

### Required Hardware

- RTX 3060 12 GB (you have this ✅)
- 16 GB RAM (you have this ✅)
- Microphone (USB or headset — any will work)
- Speakers or headphones

### Optional Hardware

- USB webcam (for vision features)

### Accounts / APIs (optional)

- **Spotify Premium** — for music control (free tier doesn't support playback API)
- **Govee account + API key** — for smart light control
- Edge-TTS requires internet (Microsoft's free neural voice service)

---

## 4. Step-by-Step Installation

### Step 1 — Install Python 3.11

1. Go to [python.org/downloads](https://www.python.org/downloads/releases/python-3119/)
2. Download **Python 3.11.x (Windows installer 64-bit)**
3. Run the installer — **check "Add Python to PATH"** at the bottom before clicking Install
4. Verify: open Command Prompt → `python --version` → should say `Python 3.11.x`

> ⚠️ **Do not use Python 3.12+ yet** — some C extension packages (dlib, certain audio libs) don't have pre-built Windows wheels for newer versions. Python 3.11 has the best compatibility.

---

### Step 2 — Install CUDA Toolkit

1. Go to [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
2. Select: Windows → x86_64 → 11 → exe (network)
3. Download and run — default install options are fine
4. Restart your PC after install
5. Verify: open Command Prompt → `nvcc --version` → should show version 12.x

---

### Step 3 — Install Ollama

1. Go to [ollama.ai](https://ollama.ai) → Download for Windows
2. Run the `.exe` installer
3. Ollama runs as a background service — no manual start needed after install
4. Verify: open Command Prompt → `ollama list` → should show an empty list (no models yet)

---

### Step 4 — Install Microsoft C++ Build Tools

This is needed to compile `dlib` (used for face recognition). Skip if you don't need face recognition.

1. Go to [visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Download the installer
3. Run it → select **"Desktop development with C++"** from the workloads list
4. Click Install — downloads ~3 GB
5. Restart when done

---

### Step 5 — Get the Sterling Files

If you're copying from another machine, put the entire `Windows` folder somewhere convenient:
```
C:\Users\YourName\Sterling\
```

Open Command Prompt and navigate there:
```cmd
cd C:\Users\YourName\Sterling
```

---

### Step 6 — Create a Virtual Environment

A virtual environment keeps Sterling's packages separate from anything else on your system.

```cmd
python -m venv ster
ster\Scripts\activate
```

Your prompt should now show `(ster)` at the start. **Always activate this before running Sterling.**

---

### Step 7 — Install PyTorch with CUDA

This is done separately from the other packages because it needs a special PyTorch CUDA build (not the default CPU one).

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

This downloads ~2.5 GB. When done, verify GPU is available:
```cmd
python -c "import torch; print(torch.cuda.is_available())"
```
Should print `True`. If it prints `False`, CUDA Toolkit is not installed correctly.

---

### Step 8 — Install Python Packages

```cmd
pip install edge-tts pyttsx3 pygame numpy pyyaml requests
pip install faster-whisper
pip install ultralytics opencv-python
pip install ollama spotipy
```

#### PyAudio (microphone access)

PyAudio requires a special install on Windows. Try in order:

**Option A — Direct pip (often works on Python 3.11):**
```cmd
pip install pyaudio
```

**Option B — Pre-built wheel:**
```cmd
pip install pipwin
pipwin install pyaudio
```

**Option C — Conda (if nothing else works):**
```cmd
conda install pyaudio
```

#### Face Recognition (optional, for knowing who's in the room)

```cmd
pip install dlib
pip install face_recognition
```

If dlib fails to compile, download a pre-built wheel from:
[github.com/z-mahmud22/Dlib_Windows_Python3.x](https://github.com/z-mahmud22/Dlib_Windows_Python3.x)

Then install:
```cmd
pip install dlib-19.24.6-cp311-cp311-win_amd64.whl
pip install face_recognition
```

---

### Step 9 — Pull the LLM

With Ollama installed, pull the recommended model:

```cmd
ollama pull qwen2.5:7b
```

This downloads ~4.4 GB. Ollama automatically uses your RTX 3060 GPU.

Verify:
```cmd
ollama run qwen2.5:7b "say hello"
```
Should respond within a couple seconds.

---

### Step 10 — Configure Sterling

Copy and edit the config file:
```cmd
copy config.yaml.example config.yaml
```

Open `config.yaml` in any text editor and update:

1. **Weather location** — find `weather:` section → change `Superior, Colorado` to your city
2. **Workspace path** — find `workspace:` → update the Windows path (or leave blank to disable)
3. **Govee** — if you don't have Govee lights, set `enabled: false`
4. **Spotify** — if you don't use Spotify, set `enabled: false`
5. **Vision** — if no webcam, set `enabled: false` under `vision:`

Everything else works out of the box.

---

### Step 11 — First Run

Make sure Ollama is running (it auto-starts as a service on Windows after install). Then:

```cmd
ster\Scripts\activate
python main.py
```

You'll see startup logs as each subsystem loads. When you see:
```
Listening for wake word...
```

Say **"Sterling"** or **"Hey Sterling"** — it should respond "Yes?" and be ready for your question.

---

### Step 12 — Enroll Your Face (optional)

For Sterling to recognize you by name:

1. Put a photo of yourself in `vision\faces\YourName.jpg` (just a clear headshot, JPG format)
2. Restart Sterling
3. When you're in front of the webcam, Sterling will identify you by name

Or do it by voice: say "Sterling" → "enroll my face as James" (it takes a snapshot).

---

## 5. Configuration

The `config.yaml` file controls everything. Key settings:

### LLM (Brain)

```yaml
llm:
  model: "qwen2.5:7b"     # the AI model — see Model Guide section below
  temperature: 0.6         # 0.0 = precise/robotic, 1.0 = creative/unpredictable
  max_tokens: 200          # max length of response (200 = a few sentences)
  context_window: 8192     # how much conversation history the LLM sees
```

### Speech-to-Text

```yaml
stt:
  model: "large-v3"        # whisper accuracy level — large-v3 is near-perfect
  device: "cuda"           # uses your RTX 3060
  compute_type: "float16"  # GPU float16 is ~10x faster than CPU int8
```

### Wake Word

```yaml
wake_word:
  phrases:
    - "hey sterling"
    - "sterling"
    - "ster"
  energy_threshold: 500    # raise if false triggers in noisy room
```

### TTS Voice

```yaml
tts:
  voice: "en-GB-RyanNeural"   # British male (Jarvis vibe)
  rate: "+5%"                 # speed
  pitch: "-5Hz"               # deeper = more serious
```

Run `edge-tts --list-voices` to see all available voices.

---

## 6. Running Sterling

### Normal Start

```cmd
cd C:\Users\YourName\Sterling
ster\Scripts\activate
python main.py
```

### Without Vision (faster startup, no webcam needed)

```cmd
python main.py --no-vision
```

### Custom Config File

```cmd
python main.py --config myconfig.yaml
```

### Stop Sterling

Say **"shut down"** or press `Ctrl+C`.

### Auto-Start with Windows (optional)

Create a `.bat` file:
```bat
@echo off
cd C:\Users\YourName\Sterling
call ster\Scripts\activate
python main.py
```

Then add a shortcut to it in your Startup folder (`Win+R` → `shell:startup`).

---

## 7. Voice Commands

### Wake Word
Say any of these to activate Sterling:
- **"Sterling"**
- **"Hey Sterling"**
- **"Ster"**

### Built-in Commands

| Say | What happens |
|---|---|
| "shut down" / "power off" | Sterling shuts off |
| "goodbye" / "I'm done" | Graceful goodbye then sleeps |
| "clear memory" / "start fresh" | Clears current session conversation |
| "forget what you know about me" | Deletes all stored facts about you |
| "how long have we been talking" | Reports session duration |
| "run diagnostics" / "system status" | Lists which subsystems are online |

### Context-Injected Queries (uses live data)

| Say | Sterling does |
|---|---|
| "what's the weather?" | Fetches live weather data |
| "what time is it?" | Reports current time |
| "what do you see?" | Describes webcam scene |
| "who's in the room?" | Reports recognized faces |
| "where's my phone?" | Checks object tracker (last seen location) |
| "what am I holding?" | Uses YOLO + optional VLM |

### Spotify

| Say | Action |
|---|---|
| "play some Radiohead" | Searches and plays |
| "play something chill" | Plays genre/mood |
| "pause the music" | Pause |
| "skip this" | Next track |
| "turn it up" / "turn it down" | Volume |
| "what's playing?" | Now playing info |

### Smart Lights (Govee)

| Say | Action |
|---|---|
| "turn on the lights" | All lights on |
| "dim the lights" | Brightness down |
| "turn the lights red" | Color change |
| "lights off" | All off |

### Memory

| Say | Action |
|---|---|
| "remember that I prefer dark mode" | Stores as a permanent fact |
| "remember I wake up at 7am" | Stores it |
| "forget what you know about me" | Wipes facts |

---

## 8. Model Selection Guide

### LLM Models (the brain — runs via Ollama on your RTX 3060)

| Model | VRAM | Speed | Intelligence | Best For |
|---|---|---|---|---|
| `llama3.2:3b` | ~2.0 GB | ⚡⚡⚡ Very fast | ⭐⭐ Basic | Low-end hardware only |
| `mistral:7b` | ~4.1 GB | ⚡⚡ Fast | ⭐⭐⭐ Good | Fast responses, daily chat |
| **`qwen2.5:7b`** ✅ | ~4.4 GB | ⚡⚡ Fast | ⭐⭐⭐⭐ Great | **Recommended — balanced** |
| `llama3.1:8b` | ~5.5 GB | ⚡⚡ Fast | ⭐⭐⭐⭐ Great | Best instruction following |
| `qwen2.5:14b` | ~8.7 GB | ⚡ Moderate | ⭐⭐⭐⭐⭐ Excellent | Smarter, still fits 3060 |
| `llama3.1:70b` | ~40+ GB | ❌ Won't fit | ⭐⭐⭐⭐⭐ Top tier | Needs multiple GPUs |

**Recommendation for your RTX 3060 12 GB:** Start with `qwen2.5:7b`. If you want more intelligence and don't mind slightly slower responses, try `qwen2.5:14b` — it fits but will be tight if vision is also running. Pull it with `ollama pull qwen2.5:14b`.

### STT Models (speech-to-text — runs via Faster-Whisper on your RTX 3060)

| Model | VRAM | Speed | Accuracy | Best For |
|---|---|---|---|---|
| `tiny` | ~150 MB | ⚡⚡⚡ Instant | ⭐⭐ OK | Wake word only |
| `base` | ~290 MB | ⚡⚡⚡ Very fast | ⭐⭐⭐ Good | Low-end hardware |
| `small` | ~490 MB | ⚡⚡ Fast | ⭐⭐⭐⭐ Very good | Mid-range |
| **`large-v3`** ✅ | ~3.0 GB | ⚡⚡ Fast on GPU | ⭐⭐⭐⭐⭐ Near-perfect | **Recommended for 3060** |

**Recommendation:** `large-v3` is the best Whisper model and runs comfortably on the 3060. Accents, mumbling, fast speech — it handles all of it. `float16` on CUDA makes it ~10x faster than the Mac CPU build.

### TTS Models (voice output)

| Option | Quality | Offline? | Setup |
|---|---|---|---|
| **Edge-TTS** ✅ | ⭐⭐⭐⭐⭐ Neural | ❌ Needs internet | Already installed |
| **pyttsx3** (fallback) | ⭐⭐ Robotic | ✅ Yes | Already installed |
| **Kokoro-82M** | ⭐⭐⭐⭐ Near neural | ✅ Yes | See Optional Features |

**Recommendation:** Use Edge-TTS (`en-GB-RyanNeural`) — it sounds the best and Microsoft's API is free. If you want fully offline TTS at similar quality, add Kokoro-82M (see Section 10).

### YOLO Vision Models

| Model | Speed | Accuracy | Best For |
|---|---|---|---|
| `yolov8n` (nano) | ⚡⚡⚡ Fastest | ⭐⭐⭐ Good | CPU or low-end GPU |
| **`yolov8s`** ✅ | ⚡⚡ Fast | ⭐⭐⭐⭐ Better | **Recommended for 3060** |
| `yolov8m` (medium) | ⚡ Moderate | ⭐⭐⭐⭐⭐ Best | 3060 can handle it |

**Recommendation:** Upgrade to `yolov8s` in config. On the 3060 it runs at 60+ fps. Change `yolo_model: "yolov8s.pt"` in config.yaml (downloads automatically on first use).

---

## 9. RTX 3060 Performance Upgrades

These are things you can do right now to take full advantage of your hardware. Each one is independent — do any or all.

### 1. Upgrade STT to Whisper Large-v3 (already set in config)

Already done in `config.yaml`. This gives you near-perfect transcription.
If you haven't pulled the model yet, faster-whisper downloads it automatically on first use.

### 2. Upgrade LLM to a 7B+ Model

```cmd
ollama pull qwen2.5:7b
```

Already set in `config.yaml`. If you want to try bigger:
```cmd
ollama pull qwen2.5:14b
```
Then change `model: "qwen2.5:14b"` in config.yaml.

### 3. Upgrade YOLO to yolov8s

In `config.yaml`, change:
```yaml
vision:
  yolo_model: "yolov8s.pt"
```
It downloads automatically. Noticeably better object detection.

### 4. Enable Vision-Language Model (VLM)

With VLM enabled, Sterling can actually *understand* what the camera sees — not just list object labels. "What am I holding?" will give a real answer.

```cmd
ollama pull moondream
```

In `config.yaml`:
```yaml
vision:
  vlm_model: "moondream"
```

moondream is ~1.7 GB VRAM. Fits easily alongside everything else on the 3060.

### 5. Enable Semantic Memory (ChromaDB)

Better long-term memory — semantic search over your conversation history instead of keyword matching.

```cmd
pip install chromadb sentence-transformers
```

In `config.yaml`:
```yaml
memory:
  chroma_enabled: true
```

### 6. Larger Context Window

The 3060 with a 7B model can handle a bigger context window (more conversation history visible to the LLM):

```yaml
llm:
  context_window: 16384    # doubled from default 8192
```

### 7. Enable Gesture Control

Wave to wake Sterling, point right/left to skip tracks:

```yaml
gestures:
  enabled: true
```

YOLOv8n-pose (~7 MB) downloads automatically.

### 8. Enable Presence Detection

Sterling notices when you walk into the room:

```yaml
presence:
  enabled: true
  greet_on_enter: true
  lights_on_enter: true
```

---

## 10. Optional Features

### Offline TTS — Kokoro-82M

High-quality offline TTS that rivals Edge-TTS. Runs fast on the 3060.

```cmd
pip install kokoro soundfile
```

Then you need to wire Kokoro into `core/tts.py` — the hooks are already planned in the codebase. Currently, Edge-TTS is the primary with pyttsx3 as fallback. Kokoro support is coming in the next build iteration.

For now: Edge-TTS works great unless you're offline.

### ChromaDB Semantic Memory

```cmd
pip install chromadb sentence-transformers
```

Enable in config:
```yaml
memory:
  chroma_enabled: true
```

This replaces keyword recall with proper semantic search. When you ask about something you discussed a week ago, it retrieves relevant context even if you use different words.

### Windows Service (Always-On)

Run Sterling as a Windows service that starts automatically at boot:

```cmd
pip install pywin32
```

See `FUTURE_ITERATIONS.md` for the service wrapper plan. For now, adding a startup `.bat` file is the simplest approach (see Section 6).

### Multiple Cameras

Change `device_index` in config to switch cameras:
```yaml
vision:
  device_index: 1    # 0 = first cam, 1 = second cam
```

---

## 11. Troubleshooting

### "torch.cuda.is_available() returns False"

PyTorch didn't install the CUDA build. Re-run:
```cmd
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Make sure CUDA Toolkit 12.x is installed first. Check with `nvcc --version`.

### "No module named 'pyaudio'"

Pre-built wheel approach:
```cmd
pip install pipwin
pipwin install pyaudio
```

Or download the wheel manually from [lfd.uci.edu](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

### "Model not found" / Ollama errors

Make sure Ollama is running:
```cmd
ollama serve
```

Pull the model if missing:
```cmd
ollama pull qwen2.5:7b
```

Check what's installed:
```cmd
ollama list
```

### Sterling doesn't hear the wake word

1. Check your mic is set as default in Windows Sound Settings
2. Raise `energy_threshold` in config if background noise is triggering false positives
3. Lower it if Sterling misses quiet speech
4. Try a USB mic — they tend to have better pickup than built-in laptop mics

### "pygame not available" error

```cmd
pip install pygame
```

If that fails:
```cmd
pip install pygame --pre
```

### STT is slow / laggy

Check that CUDA is being used:
- `device: "cuda"` in config.yaml under `stt:`
- `compute_type: "float16"` (not int8)
- Run `python -c "import torch; print(torch.cuda.is_available())"` — must print `True`

### Edge-TTS fails / no sound

1. Check internet connection (Edge-TTS needs it)
2. Check speakers are working
3. Try: `python -c "import pygame; pygame.mixer.init(); print('OK')"`

### face_recognition / dlib won't install

Install the pre-built wheel:
1. Go to [github.com/z-mahmud22/Dlib_Windows_Python3.x](https://github.com/z-mahmud22/Dlib_Windows_Python3.x)
2. Download the `.whl` for Python 3.11 (`cp311`)
3. `pip install dlib-19.x.x-cp311-cp311-win_amd64.whl`
4. `pip install face_recognition`

If you still can't get it working, set `face_recognition: false` in config.yaml — everything else still works.

### Spotify "redirect URI mismatch"

In your Spotify developer dashboard, make sure the redirect URI is exactly:
```
http://127.0.0.1:8888/callback
```
(matches what's in config.yaml)

---

## 12. How the Code Is Organized

```
main.py                     ← Entry point. Orchestrates everything.
                              Run with: python main.py

config.yaml                 ← All settings. Edit this, not the code.
config.yaml.example         ← Template — safe to share (no secrets)

core/
  wake_word.py              ← Listens for "Sterling" using Whisper tiny
  stt.py                    ← Speech → text (Whisper large-v3 on CUDA)
  llm.py                    ← Sends text to Ollama, gets AI response
  tts.py                    ← Text → speech (Edge-TTS + pygame playback)
  memory.py                 ← Session window + archive + facts store
  govee.py                  ← Controls Govee smart lights via cloud API
  spotify.py                ← Controls Spotify via Web API
  workspace.py              ← LLM-powered project scaffolding

vision/
  webcam.py                 ← Webcam + YOLO + face recognition + scene description
  object_tracker.py         ← "Where's my phone?" — tracks last-seen objects
  gesture.py                ← Wave/point gesture detection via YOLOv8-pose
  faces/                    ← Drop named JPGs here to enroll faces

utils/
  audio.py                  ← Microphone recording + VAD (voice activity detection)
  weather.py                ← Live weather via wttr.in (no API key needed)
  text.py                   ← Text cleaning for TTS
  logger.py                 ← Shared logging setup

prompts/
  system_prompt.txt         ← Sterling's personality and rules

scripts/
  discover_govee.py         ← Helper to find your Govee device IDs

memory.json                 ← Conversation archive (auto-created)
facts.json                  ← Durable facts you've told Sterling (auto-created)
object_tracker.json         ← Last-seen object positions (auto-created)
sterling.log                ← Log file (auto-created)
```

### Startup sequence (what happens when you run `python main.py`)

1. Load `config.yaml`
2. Load `prompts/system_prompt.txt` (Sterling's personality)
3. Initialize Memory → Wake Word → Recorder → STT → LLM → TTS → Vision → Govee → Spotify
4. Open one shared microphone stream
5. Start the wake word detection loop
6. Speak the startup message
7. Wait for you to say "Sterling"

---

## Quick Reference Card

```
START:      ster\Scripts\activate  →  python main.py
WAKE:       "Sterling" or "Hey Sterling"
STOP:       "shut down"  or  Ctrl+C

MUSIC:      "play [artist/song/genre]" / "pause" / "skip" / "volume up/down"
LIGHTS:     "lights on/off" / "dim the lights" / "lights [color]"
WEATHER:    "what's the weather?"
TIME:       "what time is it?"
CAMERA:     "what do you see?" / "who's in the room?"
MEMORY:     "remember that [fact]" / "forget what you know about me"
STATUS:     "run diagnostics" / "how long have we been talking?"
```

---

*Sterling Windows Build — RTX 3060 12 GB / 16 GB RAM*
*Based on the M1 Mac v1 foundation — all core architecture identical*
