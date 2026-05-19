# STERLING — Future Platform Iterations
### *Windows (RTX 3060) · Jetson / Pi · Linux · Ideas & Roadmap*

> This document outlines planned builds of Sterling for other platforms, plus a running list of
> ideas for what Sterling could actually *do*. The M1 Mac build (v1) is the foundation.

---

## Table of Contents
1. [Platform Comparison Overview](#1-platform-comparison-overview)
2. [Sterling v2 — Windows (RTX 3060 / 16 GB RAM)](#2-sterling-v2--windows-rtx-3060--16-gb-ram)
3. [Sterling v3 — Dedicated Hardware (Jetson / Pi)](#3-sterling-v3--dedicated-hardware-jetson--pi)
4. [Sterling v4 — Linux Build](#4-sterling-v4--linux-build)
5. [Shared Feature Roadmap](#5-shared-feature-roadmap)
6. [Ideas — What Could Sterling Actually Do?](#6-ideas--what-could-sterling-actually-do)
7. [Model Upgrade Path](#7-model-upgrade-path)

---

## 1. Platform Comparison Overview

| Feature | v1 — M1 Mac (8 GB) | v2 — Windows (3060) | v3 — Jetson Orin | v4 — Linux |
|---|---|---|---|---|
| **LLM Model** | Llama 3.2 3B | Llama 3.1 8B | Llama 3.2 3B–8B | Llama 3.1 8B–70B |
| **LLM Backend** | Metal | CUDA | CUDA (Jetson) | CUDA / ROCm / CPU |
| **STT Model** | Whisper Base | Whisper Large-v3 | Whisper Base/Small | Whisper Large-v3 |
| **TTS** | Edge-TTS + `say` | Kokoro (offline) | **Piper** (offline) | Piper / Coqui XTTS |
| **TTS Offline** | Partial | ✅ Full | ✅ Full | ✅ Full |
| **Vision** | HuskyLens2 | HuskyLens2 + YOLO | HuskyLens2 + YOLO | HuskyLens2 + YOLO |
| **Long-term Memory** | Planned | ChromaDB | ChromaDB | ChromaDB |
| **Home Automation** | Govee ✅ | Govee + more | Govee + more | Home Assistant |
| **Always-On** | Manual | Windows Service | systemd (24/7) | systemd / Docker |

---

## 2. Sterling v2 — Windows (RTX 3060 / 16 GB RAM)

### Vision for This Build

The RTX 3060 (12 GB VRAM) fundamentally changes what Sterling can do. CUDA acceleration means:

- **Larger LLMs** — Llama 3.1 8B or even 70B (Q4 quantized) running fast
- **Better STT** — Whisper Large-v3 for near-perfect transcription
- **Fully offline TTS** — Kokoro runs locally at Edge-TTS quality or better
- **Real-time vision** — YOLO v8/v11 on GPU alongside HuskyLens2

### Upgraded Stack

| Layer | v1 M1 Mac | v2 Windows 3060 |
|---|---|---|
| LLM | Llama 3.2 3B | Llama 3.1 8B (CUDA) |
| STT | Whisper Base (CPU) | Whisper Large-v3 (CUDA) |
| TTS | Edge-TTS (online) | Kokoro-82M (offline) |
| Vision | HuskyLens2 | HuskyLens2 + YOLO webcam |
| Memory | Session JSON | ChromaDB long-term |

### VRAM Budget
```
Llama 3.1 8B (Q4)        ~5.5 GB
Whisper Large-v3         ~3.0 GB
Kokoro-82M               ~0.3 GB
OS + buffer              ~3.0 GB
─────────────────────────────────
Total                   ~11.8 GB / 12 GB ✅
```

### Kokoro TTS
Kokoro-82M is a state-of-the-art lightweight TTS model. On a 3060 it synthesises speech faster
than real-time. Voice `bm_george` is a British male — solid Jarvis feel.

```bash
pip install kokoro soundfile
```

---

## 3. Sterling v3 — Dedicated Hardware (Jetson / Pi)

### The Vision

This is the real Jarvis setup — a dedicated box that lives in the room, runs 24/7, and has
nothing else on it. No sharing resources with a dev machine, no waking a laptop to talk to it.
Just Sterling, always on, always listening, in the corner of the room.

### Hardware Options

**Recommended: NVIDIA Jetson Orin Nano / NX**
| Component | Spec |
|---|---|
| Device | Jetson Orin Nano 8GB or Orin NX 16GB |
| CPU | 6–8 core ARM Cortex-A78AE |
| GPU | Ampere — 1024 CUDA cores (Nano) / 2048 (NX) |
| RAM | 8–16 GB LPDDR5 (shared CPU/GPU) |
| OS | JetPack 6 (Ubuntu 22.04) |
| Power | 10–25W — runs on a wall plug indefinitely |

Why Jetson over Pi:
- Has an actual GPU — CUDA acceleration for LLM, STT, and YOLO
- Runs Llama 3.2 3B comfortably in real time
- Designed for always-on AI workloads
- Supports proper CUDA builds of Ollama, faster-whisper, PyTorch

**Budget Option: Raspberry Pi 5 (8 GB)**

Pi 5 can handle Sterling but it's tight. Use it as a satellite node (wake word + audio capture
only) pointing at a more powerful central machine, not as a standalone brain.

| What works on Pi 5 | What doesn't |
|---|---|
| Wake word detection | Llama 3.2 3B (too slow) |
| Faster-Whisper tiny | Whisper base/small (slow) |
| Piper TTS | Edge-TTS (needs internet, fine actually) |
| Govee control | YOLO real-time vision |

### TTS on Jetson/Pi — Piper

Edge TTS requires an internet connection. For a dedicated always-on box you want fully offline
TTS. **Piper** is the right choice here:

- Lightweight — runs fast even on CPU, excellent on Jetson GPU
- Good quality — not quite Edge TTS but genuinely close with the right voice model
- Fully offline — no Microsoft servers, works on a plane
- ARM native builds available

```bash
pip install piper-tts
```

Best voices for Sterling's Jarvis feel:
- `en_GB-alan-medium` — British male, clean
- `en_GB-vctk-medium` — British male, natural
- `en_US-ryan-high` — American, clear and sharp

Sterling's TTS module already has a fallback architecture — adding a `PiperTTS` class in
`core/tts.py` and pointing config at it is all it would take.

### Jetson Setup Notes
```bash
# Ollama has an ARM/Jetson build
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2

# faster-whisper works natively on JetPack
pip install faster-whisper

# Piper TTS
pip install piper-tts

# Run Sterling as a systemd service (starts on boot)
sudo cp sterling.service /etc/systemd/system/
sudo systemctl enable sterling
sudo systemctl start sterling
```

### systemd Service
```ini
[Unit]
Description=Sterling AI Assistant
After=network.target ollama.service

[Service]
Type=simple
User=sterling
WorkingDirectory=/opt/sterling
ExecStart=/opt/sterling/ster/bin/python main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Sterling starts automatically on boot, restarts itself if it crashes, and runs silently in
the background forever.

---

## 4. Sterling v4 — Linux Build

Full Linux build targeting a workstation or mini-PC. Covers:

- systemd service + Docker packaging
- Home Assistant integration (MQTT/REST bridge)
- ROCm support for AMD GPUs
- Multi-room satellite mode (central server + Pi 5 nodes)
- OpenWakeWord (free, no Picovoice key)
- Coqui XTTS for highest-quality offline TTS + voice cloning

### Multi-Room Satellite Mode
```
[Pi 5 — Bedroom]  ─┐
[Pi 5 — Kitchen]  ─┤──▶  [Sterling Server]  ◀──▶  [Ollama / GPU]
[Pi 5 — Office]   ─┘
```
Each satellite handles wake word + audio only. STT, LLM, and TTS happen on the central box.

---

## 5. Shared Feature Roadmap

Features planned across all builds:

- **ChromaDB long-term memory** — "Sterling, remind me what we discussed about the API project"
- **Wake word interruption** ✅ — Saying wake word mid-response stops TTS immediately; full Whisper phrase match (not just energy); partial response discarded from memory
- **Conversation wind-down** ✅ — "I'm done" / "goodbye" / "talk later" → Sterling says goodbye → returns to wake word (not full shutdown)
- **Govee smart lights** ✅ — Cloud API, on/off/color/brightness, mid-conversation control
- **Spotify control** ✅ — Play/pause/skip/volume/search, intent parsed pre-LLM
- **Weather** ✅ — wttr.in, current + two-day forecast, any location on the fly
- **Custom wake word** — Train "Hey Sterling" via OpenWakeWord (free) or Picovoice Console
- **Multi-user voice profiles** — Identify speaker, per-user memory and greetings
- **Proactive awareness** — Sterling initiates: "Build finished — 3 warnings", "90 min break reminder"
- **Local web search** — SearXNG self-hosted, results injected into LLM context before responding
- **Smart display output** — Secondary screen with waveform, code blocks, vision feed, status HUD

---

## 6. Ideas — What Could Sterling Actually Do?

This is the fun section. The "cool, I built this — now what?" problem. Roughly ordered from
easy to build → ambitious.

---

### Daily Life

**Morning Briefing**
Sterling reads you a morning summary when you say "good morning" or walk in:
- Current time and day
- Weather (wttr.in API — no key needed)
- Any reminders set the night before
- Optional: top RSS headlines from a feed you choose

**Voice Notes**
"Sterling, note down..." → appends a timestamped entry to `notes.md`.
"Sterling, what did I note yesterday?" → reads back recent entries.
Simple file I/O, no database needed.

**Pomodoro / Focus Timer**
"Sterling, 25 minute focus session" → sets a timer, tells you when it's up.
"Sterling, how much time left?" → status update mid-session.
Good for staying on track without picking up your phone.

**Reminders**
"Sterling, remind me at 3pm to check the deployment."
Basic scheduled task using Python's `sched` module or `APScheduler`.

---

### Developer / Project Assistant

**Terminal Command Runner**
"Sterling, run the test suite" → runs a pre-configured shell command, reads back the output
summary. Sandboxed subprocess. Useful for "did it pass?" without switching windows.

**Read the Error Log**
"Sterling, what does the log say?" → reads last N lines of a log file, passes to LLM to explain.
"Sterling, anything interesting in sterling.log?" works out of the box with the existing log.

**GitHub Status**
"Sterling, any open PRs?" or "Sterling, did the build pass?" → GitHub API call, natural language
response. Needs a GitHub token in config.

**Clipboard Assistant**
"Sterling, explain what's in my clipboard" → reads clipboard content, passes to LLM.
Dead simple with `pyperclip`. Paste an error, ask Sterling what it means.

**Code Review Partner**
Point Sterling at a file: "Sterling, review auth.py" → reads the file, sends to LLM for a
quick code review. Good for a second pair of eyes before committing.

---

### Home & Environment

**Spotify / Music Control** ✅
Live — play/pause/skip/volume/search by artist or song. Needs client credentials + Premium account.

**Govee Smart Lights** ✅
Live — cloud API, on/off/color (15 colors)/brightness. Both devices configured (Bed lights + TV Backlight).

**Smart Plug Control**
TP-Link Kasa plugs have a local LAN API (no cloud required — same approach as Govee).
"Sterling, turn off the PC monitors" or "Sterling, is the soldering iron on?"

**Weather Awareness** ✅
Live via wttr.in — current conditions + two-day forecast with precip %, any location on the fly.

**Security / Presence Mode**
When you leave: "Sterling, watch the room." → HuskyLens2 / YOLO monitors for motion/faces.
Sends a desktop notification or speaks an alert if someone enters.
"Sterling, who came in while I was gone?" → reviews the detection log.

**Environmental Sensors (Jetson/Pi GPIO)**
Hook up a cheap DHT22 or BME280 sensor to GPIO pins.
"Sterling, what's the temperature in here?" → reads sensor directly.
"Sterling, humidity is high" → proactive alert when threshold crossed.

---

### Fun / Creative

**Debate Partner**
"Sterling, argue the other side of [topic]." Sterling takes the opposing position and pushes
back on everything you say. Good for stress-testing ideas.

**Daily Standup**
"Sterling, standup." → Sterling asks: what did you do yesterday, what are you doing today,
any blockers? Logs your answers to a standup file. Useful for solo devs staying accountable.

**Explain Like I'm Five**
"Sterling, ELI5 how TLS handshakes work." Forces Sterling to simplify. Good for learning
new concepts quickly without reading a whole article.

**Language Practice**
Set a mode: "Sterling, let's speak in Spanish for a bit." Sterling switches language for the
conversation. Immersive and low-stakes.

**Brainstorm Session**
"Sterling, let's brainstorm names for this project" or "Sterling, what are 10 ways I could
monetise this?" Just an LLM with context — but having it spoken out loud in the room feels
different from typing into a browser.

---

### Ambitious / Long-Term

**Agentic Task Running**
Give Sterling a goal and let it break it down and execute steps:
"Sterling, set up a new Python project called 'tracker' with a venv and FastAPI."
→ runs the shell commands, confirms each step. Needs careful sandboxing.

**Screen Awareness**
Take a screenshot, send to a vision LLM (LLaVA via Ollama):
"Sterling, what's on my screen?" or "Sterling, explain this error on my screen."
LLaVA runs locally via Ollama — `ollama pull llava`.

**Email Drafting**
Connect to Gmail/IMAP: "Sterling, draft a reply to the last email from Tom."
Sterling reads the email thread, drafts a reply, reads it back, asks if you want to send.

**Meeting Companion**
When on a call: "Sterling, take notes." → transcribes audio in the background via Whisper,
summarises at the end. "Sterling, what did we agree on?" → LLM summary of the transcript.

**Ollama Model Switcher**
"Sterling, switch to the big model" → changes to Llama 3.1 8B for a complex task,
"Switch back to fast mode" → back to 3B. Config hot-reload without restarting Sterling.

**Voice-Controlled Git**
"Sterling, commit everything with message 'fix auth bug'" → runs `git add -A && git commit -m "..."`
"Sterling, what changed since yesterday?" → `git log --since="yesterday"` summarised by LLM.

---

## 7. Model Upgrade Path

```
Platform              │  LLM                  │  STT              │  TTS
──────────────────────┼───────────────────────┼───────────────────┼───────────────────
v1 M1 Mac 8GB         │  Llama 3.2 3B (Metal) │  Whisper Base     │  Edge-TTS (online)
v2 Win RTX 3060 16GB  │  Llama 3.1 8B (CUDA)  │  Whisper Large-v3 │  Kokoro (offline)
v3 Jetson Orin        │  Llama 3.2 3B (CUDA)  │  Whisper Base     │  Piper (offline)
v4 Linux workstation  │  Llama 3.1 70B (CUDA) │  Whisper Large-v3 │  Coqui XTTS (offline)
Future                │  Local GPT-4 class    │  Parakeet-TDT     │  Voice cloning
```

---

*Updated May 2026 — M1 Mac v1 operational with lights, Spotify, weather, and wake word interruption. Jetson Orin Nano owned — migration planned.*
**Sterling Project**
