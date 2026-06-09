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
7. [Feature Backlog — v1 Mac Additions](#7-feature-backlog--v1-mac-additions)
8. [Vision Roadmap — Medium Term & Detailed View](#8-vision-roadmap--medium-term--detailed-view)
7. [Vision Upgrade — USB Webcam + YOLO + Face Recognition](#7-vision-upgrade--usb-webcam--yolo--face-recognition)
8. [Model Upgrade Path](#8-model-upgrade-path)
9. [Remote Access — iPhone / Mobile (Long-term)](#9-remote-access--iphone--mobile-long-term)

---

## 1. Platform Comparison Overview

| Feature | v1 — M1 Mac (8 GB) | v2 — Windows (3060) | v3 — Jetson Orin | v4 — Linux |
|---|---|---|---|---|
| **LLM Model** | Llama 3.2 3B | Llama 3.1 8B | Llama 3.2 3B–8B | Llama 3.1 8B–70B |
| **LLM Backend** | Metal | CUDA | CUDA (Jetson) | CUDA / ROCm / CPU |
| **STT Model** | Whisper Base | Whisper Large-v3 | Whisper Base/Small | Whisper Large-v3 |
| **TTS** | Edge-TTS + `say` | Kokoro (offline) | **Piper** (offline) | Piper / Coqui XTTS |
| **TTS Offline** | Partial | ✅ Full | ✅ Full | ✅ Full |
| **Vision** | Webcam (YOLO) | Webcam + YOLO | Webcam + YOLO | Webcam + YOLO |
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
- **Real-time vision** — YOLO v8/v11 on GPU alongside Webcam (YOLO)

### Upgraded Stack

| Layer | v1 M1 Mac | v2 Windows 3060 |
|---|---|---|
| LLM | Llama 3.2 3B | Llama 3.1 8B (CUDA) |
| STT | Whisper Base (CPU) | Whisper Large-v3 (CUDA) |
| TTS | Edge-TTS (online) | Kokoro-82M (offline) |
| Vision | Webcam (YOLO) | Webcam + YOLO webcam |
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
When you leave: "Sterling, watch the room." → Webcam (YOLO) / YOLO monitors for motion/faces.
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

## 7. Vision Upgrade — USB Webcam + YOLO + Face Recognition

USB webcam + YOLO + face_recognition. Now the live implementation in `vision/webcam.py`.
This section documents the full feature and platform-specific setup notes.

---

### Architecture

The webcam runs a background capture thread keeping the latest frame in memory so queries are instant:

```
Background thread:  cv2.VideoCapture → latest_frame (always fresh)
On query:           YOLO(latest_frame)  → detected objects + persons
                    face_recognition()  → who is in frame
```

The new `core/vision_webcam.py` will expose the same interface as `core/vision.py`
so `main.py` requires zero changes. A config flag picks the backend:

```yaml
vision:
  backend is now always webcam — no config needed
  device_index: 0          # USB camera index (0 = first camera)
  yolo_model: "yolov8n.pt" # nano = fastest | yolov8s = more accurate
  face_recognition: true
  known_faces_dir: "faces/" # folder of named JPEGs — faces/jtb.jpg etc.
```

---

### Face Enrollment

No physical button. Just drop a photo in the `faces/` folder:

```
faces/
  jtb.jpg          → recognised as "jtb"
  guest.jpg        → recognised as "guest"
```

Sterling loads encodings at startup. Adding a new person = drop a photo and restart.

---

### Dependencies

**M1 Mac:**
```bash
# YOLO + OpenCV
pip install ultralytics opencv-python

# face_recognition — needs dlib compiled first
brew install cmake
pip install dlib
pip install face_recognition

# YOLO uses Metal (MPS) automatically on M1 — no extra config
```

**Jetson Orin (JetPack 6):**
```bash
# OpenCV comes pre-installed with JetPack
# YOLO
pip install ultralytics

# dlib with CUDA support
pip install dlib  # JetPack provides CUDA — dlib picks it up automatically
pip install face_recognition

# YOLO uses CUDA automatically on Jetson
```

**Windows (RTX 3060):**
```bash
pip install ultralytics opencv-python
pip install dlib  # pre-built wheel available for Windows
pip install face_recognition

# YOLO uses CUDA automatically when torch+cuda is installed
```

---

### YOLO Model Size Guide

| Model | Size | Speed (M1) | Speed (Jetson Orin) | Best for |
|---|---|---|---|---|
| yolov8n | ~6 MB | Very fast | Very fast | Always-on monitoring |
| yolov8s | ~22 MB | Fast | Fast | Better accuracy |
| yolov8m | ~52 MB | Moderate | Fast (CUDA) | High accuracy |
| yolov8l | ~87 MB | Slow on CPU | Moderate | Jetson / GPU only |

Recommendation: `yolov8n` on M1 Mac and Jetson Nano, `yolov8s` on Jetson Orin / 3060.

---

### What It Unlocks

- **Person detection** — "Is anyone in the room?" with actual accuracy
- **Object recognition** — 80 COCO classes: laptop, phone, cup, book, etc.
- **Face recognition** — "Who's in the room?" identifies registered faces
- **Stranger detection** — alerts when an unrecognised face is detected
- **Activity awareness** — future: gesture recognition, posture detection
- **Always-on monitoring** — background thread means no query lag

---

### Files to Create / Modify

```
vision/webcam.py          ✅ live
vision/faces/             ✅ live — drop named photos here
main.py _init_vision()    ✅ updated
requirements_*.txt        ✅ updated
config.yaml               ✅ updated
```

---

## 8. Model Upgrade Path

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

## 9. Remote Access — iPhone / Mobile (Long-term)

Not a priority but worth noting for the future. The goal would be accessing Sterling from anywhere via iPhone — type or speak a question, get a response back.

### Recommended Approach (when ready)

**Step 1 — Flask API layer**
Add a small Flask server to Sterling exposing a `/chat` endpoint. Text in, response out. Sits on top of the existing Sterling core — no changes to the voice pipeline.

```python
# rough idea
@app.route("/chat", methods=["POST"])
def chat():
    text = request.json["message"]
    response = sterling.chat(text)   # reuses existing LLM + memory
    return {"response": response}
```

**Step 2 — Tailscale for remote access**
Free, 5 minute setup. Install on the Mac (or Jetson) and iPhone. They see each other like they're on the same network regardless of location. No port forwarding, no static IP, no security exposure.

```bash
# Mac
brew install tailscale
sudo tailscale up

# iPhone: install Tailscale from App Store, log in with same account
# Then hit http://[tailscale-ip]:5000 from Safari anywhere
```

**Step 3 — Simple web UI**
A basic HTML page served by Flask. Text input, send button, response display. Works in Safari on iPhone with no app install needed.

### Levels of Complexity

| Goal | Effort | Notes |
|---|---|---|
| Text chat via Safari | Low | Flask + Tailscale, afternoon project |
| Voice from iPhone | Medium | Browser mic needs HTTPS, noticeable latency |
| Proper iPhone app | High | Swift or React Native, full project |

### Notes
- Voice from browser requires HTTPS — use Cloudflare Tunnel or self-signed cert
- Latency on voice will be higher than local (audio round-trip over internet)
- Carries over to Jetson unchanged — same Flask API, different host IP
- Text chat is the sensible scope for a first pass

---

*Updated May 2026 — M1 Mac v1 operational with lights, Spotify, weather, and wake word interruption. Jetson Orin Nano owned — migration planned.*
**Sterling Project**

---

---

## 7. Feature Backlog — v1 Mac Additions

> Everything below builds on the current M1 Mac stack with no platform change required.
> Roughly ordered: **makes Sterling feel like Jarvis** → **genuinely useful daily** → **developer tools** → **smart home** → **fun / personality** → **quality of life**.

---

### Makes Sterling Feel Like Jarvis

**Good Morning Briefing**
Say “good morning” or walk in (presence detection) → Sterling gives you a full situation report:
current time, weather summary, any reminders set the night before, optional RSS headlines.
One command, zero screen time.
- Dependencies: `feedparser` for RSS, rest already live
- Config: `briefing.rss_feeds`, `briefing.headline_count`

**Proactive Alerts**
Sterling watches for things and tells you without being asked:
- Mac battery below a configured threshold → “Your laptop’s running low”
- You’ve been at your desk for N minutes without a break → break reminder
- Weather changes significantly mid-day (sunny → rain incoming)
- A watched background process or file changes
- Dependencies: `psutil` (battery/CPU), `watchdog` (file system)
- Config: `alerts.*` with per-alert enable/disable and thresholds

**Voice Notes + Recall**
“Sterling, note down — buy coffee beans.” → timestamped entry appended to `notes.md`.
“Sterling, what did I note this week?” → reads back recent entries via LLM summary.
- No new dependencies — pure file I/O
- `notes_file` path in config, searchable by date or keyword

**Mood-Aware Tone**
Sterling checks the time of day and adjusts tone automatically.
Late night → quieter and chill. Morning → perkier. Can be overridden by voice:
“Sterling, chill out” / “Sterling, be more upbeat.”
- Implemented as a time-based system prompt modifier
- Config: `tone.morning_start`, `tone.night_start`, named tone presets

**Conversation History Recall**
“Sterling, what were we talking about yesterday?”
Currently raw turns are recalled. A summarised-sessions approach — each session gets a
one-paragraph LLM summary stored in memory.json — would feel far more natural and use fewer tokens.
- Light LLM call at `end_session()` time
- Summary stored alongside the raw messages

---

### Genuinely Useful Every Day

**Timers and Reminders**
“Sterling, remind me in 20 minutes to check the oven.”
“Sterling, set a 5-minute timer.”
Sterling speaks the alert when it fires, even mid-conversation.
- `threading.Timer` or `sched` module — no dependencies
- Multiple simultaneous timers with labels
- Config: `timers.chime_sound` (optional audio cue)

**Pomodoro / Focus Mode**
“Sterling, 25-minute focus session.” → counts down, announces breaks, tracks session count.
“Sterling, how many pomodoros today?” → reports the tally.
“4 done — nice work.” Feels like a real productivity partner.
- Builds on the timer system above
- Config: `pomodoro.work_minutes`, `pomodoro.break_minutes`, `pomodoro.long_break_every`

**Clipboard Assistant**
“Sterling, explain what’s in my clipboard.”
“Sterling, fix the bug in my clipboard.”
Reads whatever is on your clipboard (code, error message, text) and passes it to the LLM.
No typing, no browser tab switching.
- `pip install pyperclip` — one dependency, cross-platform
- Massive quality-of-life win for a developer

**Read the Error / Explain the Log**
“Sterling, what does the log say?” → reads last N lines of a configured log file,
passes to LLM for a plain-English explanation.
“Sterling, why did it crash?” → same but focused on ERROR/EXCEPTION lines.
- Config: `logs` list of named log files
  ```yaml
  logs:
    - name: "sterling"
      path: "sterling.log"
    - name: "app"
      path: "/Users/jtb/src/myproject/app.log"
  ```

**Terminal Shortcuts**
Pre-configure named shell commands in config:
```yaml
shortcuts:
  - name: "run tests"
    command: "cd /Users/jtb/src/myproject && python -m pytest"
  - name: "deploy"
    command: "git push && ./deploy.sh"
```
“Sterling, run tests” → runs the command, reads back pass/fail summary via LLM.
- Sandboxed subprocess with timeout
- Config: `shortcuts` list with `name` + `command` pairs

**Ollama Model Switcher**
“Sterling, switch to big brain mode.” → hot-swaps to a larger model for a complex task.
“Sterling, back to fast mode.” → back to the default model.
No restart required — just changes `self._llm._model` and confirms the swap.
- Config: named model aliases
  ```yaml
  llm:
    model_aliases:
      fast: "llama3.2:3b"
      big: "llama3.1:8b"
      code: "codellama:7b"
  ```

---

### Developer Tools

**Git Voice Control**
“Sterling, what changed since yesterday?” → `git log --since=yesterday` → LLM summary.
“Sterling, commit everything with message ‘fix auth bug’.” → runs `git add -A && git commit`.
“Sterling, what branch am I on?” / “any uncommitted changes?”
- Subprocess calls to git — no dependencies
- Config: `git.repo_path`, `git.require_confirmation` for destructive ops

**GitHub Integration**
“Sterling, any open PRs?” / “Sterling, did the build pass?”
GitHub REST API — PRs, CI status, issues, notifications.
- `pip install PyGithub` or plain `httpx` calls
- Config: `github.token`, `github.default_repo`

**Code Review Partner**
“Sterling, review auth dot py.” → Sterling reads the file, sends to LLM for a quick review.
Good for a second opinion before committing. Works with any file in the workspace.
- No new dependencies
- Uses the existing code-generation LLM prompt in reverse

**System Stats**
“Sterling, what’s using the most CPU?” / “Sterling, how’s memory?”
`psutil` reads live stats — LLM gives a one-sentence summary.
“Ollama is using 4 GB, everything else is fine.”
- `pip install psutil`
- Also enables the battery alert from Proactive Alerts above

**Build / Test Watcher**
Run a build or test suite in the background. Sterling tells you when it finishes and whether it passed.
Like a CI notification, but it just talks to you.
- Background subprocess thread, stdout parsed for pass/fail signals
- Config: `watcher.success_pattern`, `watcher.failure_pattern` (regex)

---

### Smart Home & Environment

**Scene Presets**
“Sterling, movie mode.” → dims lights to warm amber, pauses music.
“Sterling, work mode.” → lights to full white, plays focus playlist.
“Sterling, gaming mode.” → RGB chaos, whatever you want.
Defined entirely in config — no code changes required:
```yaml
scenes:
  movie:
    lights: { brightness: 20, color: "warm white" }
    spotify: { action: pause }
  work:
    lights: { brightness: 100, color: "white" }
    spotify: { action: play, query: "focus music" }
```

**Do Not Disturb Mode**
“Sterling, don’t interrupt me for 30 minutes.”
Sterling won’t speak or acknowledge ambient sounds for the duration.
Still responds if directly addressed. Auto-cancels after the timer expires.
- Config: `dnd.default_minutes`

**Sleep Mode**
“Sterling, good night.” → dims lights to warm red, lowers TTS volume,
stops responding to ambient triggers, optionally sets a morning alarm.
- Combines scene presets + DND + volume control
- Config: `sleep.light_color`, `sleep.light_brightness`, `sleep.volume`

**Quiet Hours**
Time-range config where Sterling automatically lowers volume and skips proactive alerts.
So a 2am build completion doesn’t wake you up.
```yaml
quiet_hours:
  enabled: true
  start: "22:00"
  end: "08:00"
  volume_multiplier: 0.3   # 30% of normal volume
  block_proactive: true    # no unsolicited speech
```

**Energy Awareness**
Track how long lights have been on. Remind you if you’ve left a room occupied for N hours
or if lights are still on in an empty room (presence detection + Govee combined).

---

### Fun & Personality

**Sterling’s Opinions**
Let Sterling actually have preferences and express them.
“Sterling, what do you think of this approach?” → real opinion, not just a description.
The personality is already set up in the system prompt — just lean into it more.
Config: `personality.opinionated: true/false`

**Explain Like I’m 5**
“Sterling, ELI5 how OAuth works.”
Forces the LLM into maximally simple explanation mode.
Great shortcut for learning something new without reading a whole article.
- One config phrase trigger, one system prompt modifier

**Debate Mode**
“Sterling, argue the other side.” → Sterling takes the opposite position on whatever
you just said and pushes back intelligently. Good for stress-testing ideas and
checking your assumptions. Toggle off with “ok, you can stop arguing.”

**Daily Standup**
“Sterling, standup.” → asks three questions:
1. What did you do yesterday?
2. What are you doing today?
3. Any blockers?
Logs your answers with timestamps to `standup.md`.
Good for solo devs staying accountable. “4 days in a row — nice streak.”
- Config: `standup.file`, `standup.questions` (customisable list)

**“What Should I Work On?”**
Sterling looks at your notes, recent git activity, open reminders, and the time of day
and suggests a focus for the session. Basically just LLM + context injection,
but it feels like having an actual project manager.
“You’ve got a reminder about the API refactor, your last commit was 2 days ago,
and it’s a Tuesday morning — sounds like a good time to tackle that.”

**Language Practice Mode**
“Sterling, let’s speak in Spanish for a bit.”
Sterling switches language for the conversation. Immersive, low-stakes practice.
- Config: `language_practice.supported_languages`
- Auto-reverts after session or when you say “switch back to English”

**Trivia / Quiz Mode**
“Sterling, quiz me on Python.” → fires questions, tracks score, adjusts difficulty.
Good for killing 10 minutes and actually learning something.

---

### Quality of Life

**Hot-Reload Config**
“Sterling, reload config.” → re-reads `config.yaml` without restarting.
Change a phrase list, threshold, or response string and apply it live.
- Zero new dependencies
- Useful during tuning sessions

**Per-Session Personality Override**
“Sterling, be more formal today.”
“Sterling, short answers only.”
“Sterling, don’t make jokes for a bit.”
Injects a temporary modifier into the system prompt for the current session.
Resets on restart or when you say “back to normal.”

**Multi-Device Audio Output**
Route Sterling’s voice to a specific audio output device via config.
Useful when you move between desk speakers and a Bluetooth speaker in another part of the room.
- `sounddevice` or macOS `SwitchAudioSource` CLI
- Config: `audio.output_device` (device name or index)

**Web Search**
Self-hosted SearXNG or a simple DuckDuckGo instant-answer fetch.
“Sterling, search for...” → LLM gets actual current results, not just training data.
Huge for anything time-sensitive: prices, news, documentation, release notes.
- `httpx` call to SearXNG or DDG API — no heavy dependencies
- Config: `search.provider`, `search.results_count`

**Notification Mirroring**
Pull macOS notifications or app webhooks and have Sterling mention important ones.
“You got a Slack message from Tom.” / “Your GitHub Action failed.”
Filter by app, sender, or keyword so it’s not constant noise.
- `terminal-notifier` or macOS `osascript` for notification access
- Config: `notifications.apps`, `notifications.keywords`, `notifications.quiet_hours`

**Sterling Status HUD**
A small always-on terminal panel (using `rich` or a tiny Tkinter window) showing:
- Current mode (listening / thinking / speaking)
- Last thing Sterling heard
- Active timers
- Presence status
- System health (RAM, CPU, Ollama up/down)
Optional — off by default, toggle with `--hud` flag.

**Remote Access (iPhone / Anywhere)**
Flask API layer on top of Sterling + Tailscale for remote access.
Type or speak a question from your phone, get Sterling’s response back.
Full write-up in section 9 (Remote Access).

---

## 8. Vision Roadmap — Medium Term & Detailed View

### What's live now (v1)

| Feature | Status |
|---|---|
| YOLO object detection (80 COCO classes) | ✅ Live |
| Face recognition (dlib + face_recognition) | ✅ Live |
| Scene description → LLM context injection | ✅ Live |
| Expression hints (smile detection via lip landmarks) | ✅ Live |
| Activity inference from object combos | ✅ Live |
| Object tracking (last-seen position, JSON-backed) | ✅ Live |
| Face enrollment by voice ("remember this person as X") | ✅ Live |
| Presence detection (optional background monitor) | ✅ Live |
| Gesture commands — wave / hands up / point (YOLOv8-pose) | ✅ Live (opt-in) |

---

### "More Detailed View" — Vision-Language Model (VLM)

**This is the single highest-leverage upgrade for the camera.**

Currently Sterling's vision pipeline is:
```
Camera frame → YOLO → list of labels + positions → text description → LLM
```
The LLM **never sees the actual image** — it just reads a text summary like
"cell phone (centre, small — possibly in hand)."  YOLO can only report the
80 object classes it was trained on; it misses pens, papers, expressions,
text, logos, food, gestures, and anything unusual.

With a Vision-Language Model the pipeline becomes:
```
Camera frame → VLM (sees actual pixels) → natural language response
```

**Available right now via Ollama:**

| Model | Size | Quality | Notes |
|---|---|---|---|
| `moondream` | 1.7 GB | Good for simple queries | Fastest, runs fine on M1 8 GB |
| `llava:7b` | 4.1 GB | Excellent | Best balance on M1 |
| `llava:13b` | 8.0 GB | Near-GPT4V quality | Needs Jetson or 16 GB Mac |
| `llava-phi3` | 2.9 GB | Very good | Phi-3 backbone, efficient |

**How it works:**
```python
import base64, cv2, httpx

frame = vision.get_frame()
_, buf = cv2.imencode(".jpg", frame)
b64 = base64.b64encode(buf).decode()

resp = httpx.post("http://localhost:11434/api/generate", json={
    "model": "moondream",
    "prompt": "What is this person holding? Describe briefly.",
    "images": [b64],
    "stream": False,
})
answer = resp.json()["response"]
```

**What it unlocks that YOLO can't do:**
- "What am I holding?" — accurately reads text on labels, identifies specific items
- "How do I look?" — reads actual facial expression, not just lip geometry
- "What's on my screen?" — reads the actual display content
- "What's written on that?" — OCR without a separate model
- "Describe the mood in the room" — holistic understanding
- Any open-ended visual question

**Cost:** VLM inference takes 2–5 seconds on M1 (vs ~100 ms for YOLO).  Good for
on-demand queries, not continuous monitoring.  YOLO stays for background tasks;
VLM fires when the user asks a specific visual question.

**Implementation plan (medium term):**
1. `ollama pull moondream` (1.7 GB, fits in M1 8 GB alongside llama3.2:3b)
2. Add `core/vlm.py` — wraps Ollama's vision endpoint
3. In `main.py`, if `_is_vision_query()` and VLM is configured, route to VLM
   instead of YOLO scene description
4. Config flag: `vision.vlm_model: "moondream"` (empty = use YOLO)

---

### Medium-Term Vision Features

**Thumbs up / thumbs down gestures**
YOLOv8-pose only gives 17 body keypoints (wrists, elbows, shoulders etc.) — not
finger-level.  Thumb direction needs a dedicated hand-landmark model.
- **MediaPipe Hands** — 21 finger keypoints, runs well on M1 CPU.  Python 3.14
  support is limited; check `pip install mediapipe` and test.
- **YOLOv8n-pose → finger ext.** — there are fine-tuned models that add hand
  keypoints on top of body pose.
- Once finger keypoints are available: thumb pointing up above fist = thumbs up,
  below = thumbs down.

**OCR — reading text in frame**
```bash
pip install pytesseract
brew install tesseract
```
"What does that say?" → capture frame → `pytesseract.image_to_string(frame)` →
inject into LLM.  Works for books, whiteboards, screens, labels, sticky notes.

**Scene change alerts (passive monitoring)**
Background frame-differencing (cheap — no YOLO):
- Compare current frame to a reference frame every N seconds
- If pixel difference > threshold, run YOLO to classify what changed
- Alert: "Something moved on the right side of the room"
- Useful as a lightweight intruder / pet detection without continuous YOLO

**Person-specific context on recognition**
When a known face is detected, auto-pull that person's last conversation or notes
from memory.json.  "Oh hey James — last time we talked you were working on the
authentication bug."  Needs face recognition + memory integration.

**Multi-camera support**
Multiple `WebcamVision` instances (different `device_index` values).
Each reports independently; LLM gets a combined scene description.
Useful for: desk cam + doorbell cam, or room coverage.

**Depth estimation (single camera)**
MiDaS (via torch hub) can estimate relative depth from a single RGB image.
"That cup is about arm's reach away on your left."
Adds genuine spatial awareness without a depth camera.
~100 ms on M1 CPU for MiDaS-Small.

**Activity recognition (expanded)**
Current: simple object-combo heuristics.
Medium term: YOLOv8-pose skeleton + temporal analysis.
- Sitting posture + laptop = focused work
- Walking (keypoints moving) = in motion
- Leaning back = relaxed / resting
Needs storing a short sequence of frames, not just the latest.

**Timeline / visual log**
Save a timestamped thumbnail when notable events occur:
- Face recognised (save who + when)
- New object detected for the first time in a while
- Room occupancy changes
Ask "what was happening at 3pm?" → Sterling describes from the visual log.
Low storage cost: save 1 frame per event as a small JPEG.

**Screen awareness (desktop)**
`mss` (cross-platform screenshot) → send to VLM:
"Sterling, what's on my screen?" — reads the actual desktop.
"Sterling, explain this error" — you Alt-Tab to the terminal, ask Sterling.
No camera required — uses the VLM pipeline above.

---

*Vision roadmap last updated: 2026-06*
