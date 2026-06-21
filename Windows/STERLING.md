# STERLING — Advanced AI Room Assistant
### *"I am at your service."*

> Inspired by J.A.R.V.I.S. from Iron Man. Sterling is a locally-running, voice-first AI assistant
> combining a large language model, real-time speech processing, computer vision, and room awareness
> into a single cohesive system.

---

## Table of Contents
1. [Vision](#1-vision)
2. [System Architecture](#2-system-architecture)
3. [Core Components](#3-core-components)
4. [Primary Functions](#4-primary-functions)
5. [Hardware — M1 Mac (Current Build)](#5-hardware--m1-mac-current-build)
6. [Software Stack](#6-software-stack)
7. [Installation & Setup](#7-installation--setup)
8. [Configuration Reference](#8-configuration-reference)
9. [Usage Guide](#9-usage-guide)
10. [Voice Command Reference](#10-voice-command-reference)
11. [Project File Structure](#11-project-file-structure)
12. [Extending Sterling](#12-extending-sterling)
13. [Performance Notes — M1 Mac](#13-performance-notes--m1-mac)
14. [Known Limitations](#14-known-limitations)
15. [Roadmap](#15-roadmap)

---

## 1. Vision

Sterling is a **persistent, always-on room intelligence**. It listens passively, activates on a wake
word, understands natural speech, reasons with a local LLM, and responds in a refined, near-human
voice — all without sending data to the cloud.

Think less "smart speaker" and more **digital co-pilot**:

- It knows your projects and helps you move them forward.
- It observes the room through a vision module and can identify faces and objects.
- It answers questions, brainstorms ideas, keeps context across a conversation, and handles
  environmental tasks.
- Everything runs locally. Your data stays yours.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STERLING CORE LOOP                        │
│                                                                   │
│  ┌───────────────┐     Wake Word      ┌────────────────────┐    │
│  │   Microphone  │ ──────────────────▶│  Porcupine Engine  │    │
│  │   (always on) │                    │  (Wake Word Det.)  │    │
│  └───────────────┘                    └────────┬───────────┘    │
│                                                │ Triggered       │
│                                                ▼                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   CONVERSATION PIPELINE                    │  │
│  │                                                            │  │
│  │  Microphone ──▶ VAD Recorder ──▶ Faster-Whisper (STT)    │  │
│  │                                         │                  │  │
│  │                                         ▼                  │  │
│  │                               Memory / Context             │  │
│  │                                         │                  │  │
│  │                                         ▼                  │  │
│  │                            Ollama (Llama 3.2 LLM)         │  │
│  │                                         │                  │  │
│  │                                         ▼                  │  │
│  │                         Edge-TTS ──▶ afplay (Speaker)     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─────────────────────┐    USB/UART    ┌──────────────────────┐ │
│  │   USB Webcam + YOLO │◀──────────────▶│   Vision Module      │ │
│  │   (Face/Object Det.)│               │  (sterling/core/)    │ │
│  └─────────────────────┘               └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow — Single Interaction

```
[Idle: Wake Word Listen Loop]
         │
         │  "Hey Sterling" detected
         ▼
[Acknowledgement TTS: "Yes?"]
         │
         ▼
[Record until silence — VAD]
         │
         ▼
[Faster-Whisper STT → text]
         │
         ├──▶ [Special Command? → handle directly]
         │
         ▼
[Memory: append user turn]
         │
         ▼
[Ollama Llama 3.2 → streamed response]
         │
         ▼
[Memory: append assistant turn]
         │
         ▼
[Sentence-streamed Edge-TTS → Speaker]
         │
         ▼
[Return to Wake Word Listen Loop]
```

---

## 3. Core Components

### 3.1 Wake Word Detection — Porcupine (Picovoice)

| Property | Value |
|---|---|
| Library | `pvporcupine` |
| Keyword | `"jarvis"` (built-in) or custom `.ppn` file |
| Latency | < 50 ms detection |
| CPU Usage | ~2–5% (M1 core) |
| Offline | ✅ 100% local |
| Accuracy | Very high — trained neural keyword model |

Porcupine runs a dedicated audio stream at its required sample rate (16 kHz, 512-frame chunks).
When the wake word is detected, control passes to the recorder. The free tier of Picovoice requires
a free API key from [picovoice.ai](https://picovoice.ai).

**Built-in keywords available:** `alexa`, `bumblebee`, `computer`, `hey google`, `hey siri`,
`jarvis`, `ok google`, `picovoice`, `porcupine`, `terminator`.

---

### 3.2 Speech-to-Text — Faster-Whisper

| Property | Value |
|---|---|
| Library | `faster-whisper` |
| Model | `base` (74M params — fast + accurate for English) |
| Backend | CTranslate2 with INT8 quantization |
| Device | CPU (M1 Apple Silicon optimized) |
| Latency | ~0.3–0.8s for typical utterance on M1 |
| Offline | ✅ 100% local |

Faster-Whisper is a reimplementation of OpenAI Whisper using CTranslate2, giving 2–4× speedup
over the original. On M1 Mac with `int8` compute type, the `base` model offers the best
speed/accuracy tradeoff.

**Model size options:**

| Model | Params | VRAM / RAM | Speed (M1) |
|---|---|---|---|
| `tiny` | 39M | ~390 MB | Very fast |
| `base` | 74M | ~580 MB | Fast ← recommended |
| `small` | 244M | ~967 MB | Moderate |
| `medium` | 769M | ~3.1 GB | Slow on M1 |

**VAD (Voice Activity Detection):** Sterling uses energy-based VAD to detect when the user has
finished speaking. Recording stops after `silence_duration` seconds of audio below
`silence_threshold` RMS energy. This prevents the system from sending empty audio to Whisper.

---

### 3.3 Language Model — Ollama + Llama 3.2

| Property | Value |
|---|---|
| Runtime | Ollama |
| Model | `llama3.2` (3B params, 4-bit quantized) |
| Context | 4096 tokens (~3000 words) |
| Backend | Metal GPU acceleration (M1 Mac) |
| RAM Usage | ~2.5 GB |
| Offline | ✅ 100% local |
| API | Ollama HTTP API (`localhost:11434`) |

Ollama handles model management, hardware acceleration, and serves a simple REST API. On M1 Mac,
Llama 3.2 3B runs with Metal acceleration, producing tokens quickly.

**Why Llama 3.2 3B?**
- Fast enough for real-time conversation (15–30 tokens/sec on M1)
- Small enough to leave headroom for STT, TTS, and the OS on 8 GB RAM
- High quality: instruction-tuned, excellent reasoning for its size
- Supports the Jarvis-style personality system prompt well

**System Prompt:** Sterling's personality is defined in `prompts/system_prompt.txt`. It instructs
the model to behave as a sophisticated, calm, slightly formal AI assistant — knowledgeable,
precise, and occasionally dry-witted. All LLM responses are kept free of markdown to ensure
clean TTS output.

**Memory Management:** Conversation history is stored in a sliding window of the last N turns
(configurable). The system prompt is always prepended. This gives Sterling persistent context
within a session without overflowing the context window.

---

### 3.4 Text-to-Speech — Edge-TTS

| Property | Value |
|---|---|
| Library | `edge-tts` |
| Voice | `en-GB-RyanNeural` (British male — Jarvis feel) |
| Quality | Neural TTS — near human |
| Latency | ~0.5–1.2s first word |
| Requirement | Internet connection |
| Playback | macOS `afplay` (built-in) |

Edge-TTS uses Microsoft's Azure neural TTS engine via the same endpoint as Microsoft Edge's
read-aloud feature. The `en-GB-RyanNeural` voice delivers a refined British male tone that
perfectly matches the Jarvis aesthetic.

**Sentence Streaming:** Sterling doesn't wait for the full LLM response before speaking. As the
LLM streams tokens, complete sentences are batched and sent to TTS one at a time. This reduces
the perceived response latency significantly.

**Offline fallback:** If internet is unavailable, Sterling falls back to macOS `say` command
with the "Alex" system voice. This is configured automatically.

**Text cleaning:** Before TTS, Sterling strips all markdown formatting (`**`, `*`, `#`, `` ` ``),
bullet points, and special characters that would sound odd when spoken.

---

### 3.5 Vision System — USB Webcam + YOLO

| Property | Value |
|---|---|
| Hardware | Any USB webcam |
| Connection | USB Video (UVC) |
| Library | `ultralytics` (YOLO) + `face_recognition` |
| Modes | Object detection (80 COCO classes), Face recognition |
| Offline | ✅ 100% local |

Sterling uses a standard USB webcam with YOLOv8 for real-time object and person detection,
and the face_recognition library for identifying enrolled faces.

**Sterling Vision Features:**
- **Face recognition:** Identify enrolled faces and greet users by name
- **Object detection:** 80+ COCO classes — laptop, phone, cup, person, etc.
- **Presence detection:** Detect when someone enters/leaves the room
- **Stranger detection:** Unrecognised faces get labelled as "unknown"

**Vision Setup:**
1. Connect a USB webcam
2. Install: `brew install cmake && pip install dlib ultralytics opencv-python face_recognition`
3. Set `vision.enabled: true` in `config.yaml`
4. Enrol faces: drop a named photo into `vision/faces/` (e.g. `vision/faces/jtb.jpg`), restart

---

### 3.6 Memory & Context Management

Sterling maintains conversation memory at three levels:

```
┌─────────────────────────────────────────────────┐
│           STERLING MEMORY ARCHITECTURE           │
│                                                  │
│  LEVEL 1: Session Memory (in-RAM)               │
│  ├─ Last N conversation turns                   │
│  ├─ Current project context                     │
│  └─ Cleared on shutdown                         │
│                                                  │
│  LEVEL 2: Project Notes (planned)               │
│  ├─ Persistent key-value store                  │
│  ├─ Project names, goals, status                │
│  └─ Survives reboots                            │
│                                                  │
│  LEVEL 3: Long-Term Memory (planned)            │
│  ├─ ChromaDB vector store                       │
│  ├─ Semantic search over past conversations     │
│  └─ "Sterling, what were we working on Monday?" │
└─────────────────────────────────────────────────┘
```

---

## 4. Primary Functions

### 4.1 Project Assistant

Sterling is designed to be your **engineering co-pilot**. It helps with:

- **Project Planning** — Breaking down large goals into phases, tasks, and milestones
- **Architecture Design** — Suggesting system designs, data flow, component breakdowns
- **Debugging** — Walking through problems, suggesting fixes, explaining error messages
- **Code Review** — Analyzing logic, suggesting improvements
- **Research** — Explaining concepts, comparing technologies, summarizing options
- **Documentation** — Drafting READMEs, API docs, comments

*Example interactions:*
> "Sterling, I'm building a REST API in FastAPI. Help me design the authentication flow."
> "Sterling, my Python script keeps crashing with a segfault in numpy. What should I check?"
> "Sterling, compare SQLite vs PostgreSQL for a project with 50k daily users."

---

### 4.2 Conversational AI

Sterling is **easy to talk to**. It maintains context across a conversation, remembers what you
said earlier in the session, and can handle:

- Natural follow-up questions ("What about the other approach?")
- Multi-topic conversations ("Actually, forget that. Let's talk about...")
- Clarification requests ("Wait, can you explain that last part differently?")
- Casual conversation and brainstorming
- Answering general knowledge questions

---

### 4.3 Room Assistant

Leveraging the webcam and environmental awareness:

- **Greeting:** Recognizes your face and greets you when you enter the room
- **Presence:** Tracks who is in the room
- **Alerts:** Notifies of unrecognized persons (optional security mode)
- **Object ID:** "Sterling, what is this?" — hold something up to the camera
- **Ambient info:** "Sterling, is anyone in the room?" (future: temperature, light sensors)

---

### 4.4 Idea Generation & Brainstorming

- Project name generation
- Feature ideation for apps, games, hardware projects
- Business model exploration
- Creative writing prompts and story development
- Solution brainstorming for engineering challenges

---

### 4.5 Knowledge & Research Assistant

- Explain complex topics simply or in depth based on your preference
- Summarize technical papers or concepts described verbally
- Compare frameworks, languages, tools, and approaches
- Science, math, history, philosophy, technology — all on-board

---

### 4.6 Personal Productivity

- Daily briefing (time, agenda reminders — future integration)
- Voice-controlled notes and reminders (planned)
- Timer and alarm support (planned)
- Workflow guidance: "Walk me through deploying to AWS step by step"

---

## 5. Hardware — M1 Mac (Current Build)

| Component | Requirement | Notes |
|---|---|---|
| CPU | Apple M1 | Metal GPU used for LLM inference |
| RAM | 8 GB | ~5.5 GB used at peak |
| Microphone | Built-in or USB | Any standard mic works |
| Speaker | Built-in or external | `afplay` uses system default |
| USB webcam | Any UVC-compatible webcam | Optional — vision disabled without it |
| Internet | For TTS only | Edge-TTS requires network; `say` fallback is offline |
| Storage | ~3 GB free | For models (Whisper base ~580 MB, Llama 3.2 ~2.5 GB) |
| macOS | 12 Monterey+ | Required for Metal + latest Python |

**RAM Budget (approximate at runtime):**

```
macOS + background apps     ~2.0 GB
Ollama (Llama 3.2 3B 4-bit) ~2.5 GB
Faster-Whisper (base)       ~0.6 GB
Sterling Python process      ~0.3 GB
Headroom                    ~2.6 GB
─────────────────────────────────────
Total                       ~8.0 GB ✅ (tight but manageable)
```

---

## 6. Software Stack

| Layer | Technology | Version |
|---|---|---|
| LLM Runtime | Ollama | Latest |
| LLM Model | Llama 3.2 | 3B (4-bit quantized) |
| STT | Faster-Whisper | ≥ 1.0.0 |
| Wake Word | pvporcupine | ≥ 3.0.0 |
| TTS | edge-tts | ≥ 6.1.0 |
| TTS Fallback | macOS `say` | Built-in |
| TTS Playback | macOS `afplay` | Built-in |
| Vision | pyserial | ≥ 3.5 |
| Audio I/O | PyAudio | ≥ 0.2.14 |
| Config | PyYAML | ≥ 6.0.0 |
| Language | Python | 3.11+ |
| Package Env | venv (`ster/`) | — |

---

## 7. Installation & Setup

### Prerequisites

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3.11+:**
   ```bash
   brew install python@3.11
   ```

3. **Install PortAudio** (required by PyAudio):
   ```bash
   brew install portaudio
   ```

4. **Install Ollama:**
   ```bash
   brew install ollama
   # Or download from https://ollama.ai
   ```

5. **Pull the Llama 3.2 model:**
   ```bash
   ollama pull llama3.2
   ```

6. **Get a free Picovoice Access Key:**
   - Sign up at [picovoice.ai](https://picovoice.ai)
   - Copy your access key from the dashboard
   - Paste it into `config.yaml` under `porcupine.access_key`

---

### Installation Steps

```bash
# Clone / navigate to project
cd /path/to/sterling

# Activate virtual environment (already created as ster/)
source ster/bin/activate

# Install all dependencies
pip install -r requirements_mac.txt

# Verify Ollama is running
ollama serve &
ollama list  # should show llama3.2

# Configure Sterling
cp config.yaml.example config.yaml
# Edit config.yaml — add your Picovoice key and adjust settings

# Run Sterling
python main.py
```

---

### Vision Setup (Webcam + YOLO)

1. Connect a USB webcam
2. Install dependencies:
   ```bash
   brew install cmake
   pip install dlib ultralytics opencv-python face_recognition
   ```
3. Set `vision.enabled: true` in `config.yaml`
4. To enrol faces, drop a named photo into `vision/faces/`:
   ```
   vision/faces/jtb.jpg   → recognised as "jtb"
   ```
5. Restart Sterling — face encodings load at startup

---

## 8. Configuration Reference

**`config.yaml`:**

```yaml
sterling:
  name: "Sterling"
  startup_message: "Sterling online. How may I assist you?"

porcupine:
  access_key: "YOUR_KEY_HERE"   # Free from picovoice.ai
  keyword: "jarvis"              # Built-in: jarvis, porcupine, computer, etc.
  # keyword_path: "custom.ppn"  # Path to custom .ppn file (optional)

llm:
  model: "llama3.2"
  base_url: "http://localhost:11434"
  temperature: 0.7
  context_window: 4096
  stream: true

stt:
  model: "base"         # tiny | base | small
  language: "en"
  device: "cpu"
  compute_type: "int8"

tts:
  voice: "en-GB-RyanNeural"   # Edge-TTS voice
  rate: "+5%"                  # Speech rate adjustment
  pitch: "-5Hz"                # Pitch adjustment (lower = deeper)
  fallback_voice: "Alex"       # macOS say fallback voice

audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 1024
  silence_threshold: 500       # RMS energy; adjust for your mic
  silence_duration: 1.5        # Seconds of silence before stopping

vision:
  enabled: true
  port: null                   # null = auto-detect
  baud_rate: 9600
  known_faces_dir: "vision/faces"  # Drop named photos here to enrol faces
  confidence_thresh: 0.45

memory:
  max_history: 20              # Max conversation turns to keep

logging:
  level: "INFO"                # DEBUG | INFO | WARNING | ERROR
  file: "sterling.log"
```

---

## 9. Usage Guide

### Starting Sterling

```bash
source ster/bin/activate
ollama serve &       # Make sure Ollama is running
python main.py
```

Sterling will say **"Sterling online. How may I assist you?"** and begin listening.

---

### Basic Interaction

1. Say **"Hey Sterling"** (or your configured wake word)
2. Wait for the acknowledgement tone / **"Yes?"**
3. Speak your request naturally
4. Sterling will respond via voice

---

### Tips for Best Results

- Speak clearly after the acknowledgement — wait ~0.5s before talking
- You can speak at a normal pace — Whisper handles natural speech well
- Keep individual requests reasonably focused — follow-up in the same conversation
- If Sterling mishears, just say the wake word again and rephrase
- For technical topics, be specific: "using Python 3.11" vs just "in Python"
- You can interrupt a project discussion and return to it: Sterling maintains session context

---

## 10. Voice Command Reference

| Command | What Sterling Does |
|---|---|
| "Hey Sterling" | Activates — waits for your request |
| "Clear memory" / "Forget everything" | Wipes conversation history |
| "Goodbye" / "Shut down" / "Power off" | Graceful shutdown |
| "What do you see?" | Reports vision module observations |
| "Who's in the room?" | Reports detected faces (if vision enabled) |
| "What is this?" | Identifies object in webcam frame |
| "That's all, thank you" | Closes conversation gracefully |

All other speech is processed as a natural language request to the LLM.

---

## 11. Project File Structure

```
sterling/
│
├── STERLING.md                  ← This document
├── FUTURE_ITERATIONS.md         ← Windows & Linux platform plans
├── README.md                    ← Quick-start guide
│
├── main.py                      ← Entry point — orchestrates all components
├── config.yaml                  ← Runtime configuration
├── config.yaml.example          ← Example config (safe to share)
├── requirements_mac.txt         ← Python dependencies for M1 Mac
├── setup_mac.sh                 ← Automated setup script
│
├── core/                        ← Core component modules
│   ├── __init__.py
│   ├── wake_word.py             ← Porcupine wake word detector
│   ├── stt.py                   ← Faster-Whisper speech-to-text
│   ├── llm.py                   ← Ollama LLM client
│   ├── tts.py                   ← Edge-TTS + macOS say fallback
│   ├── vision/webcam.py         ← USB webcam + YOLO + face_recognition
│   └── memory.py                ← Conversation context management
│
├── utils/                       ← Utility modules
│   ├── __init__.py
│   ├── audio.py                 ← Microphone recording + VAD
│   ├── text.py                  ← Text cleaning for TTS
│   └── logger.py                ← Logging setup
│
├── prompts/
│   └── system_prompt.txt        ← Sterling personality + instructions
│
└── ster/                        ← Python virtual environment (don't commit)
```

---

## 12. Extending Sterling

### Adding Custom Wake Words

Picovoice offers a free custom wake word trainer at [picovoice.ai/console](https://picovoice.ai/console).
Train "Hey Sterling", "Sterling", or any phrase, download the `.ppn` file, and point `config.yaml`
at it:

```yaml
porcupine:
  keyword_path: "assets/Hey-Sterling_en_mac_v3_0_0.ppn"
```

---

### Adding New Voice Commands

In `main.py`, add entries to the `_handle_command()` method:

```python
if "set a timer" in text_lower:
    # parse duration, start timer thread
    return True
```

---

### Adding Long-Term Memory (ChromaDB)

Install ChromaDB:
```bash
pip install chromadb sentence-transformers
```

Then extend `core/memory.py` with a `VectorMemory` class that embeds and retrieves past
conversations semantically. Sterling can then answer "What were we working on last Tuesday?"

---

### Adding Home Automation

Integrate with:
- **HomeKit** via `homekit` Python library
- **Home Assistant** REST API
- **Phillips Hue** API

```python
# Example: "Sterling, turn off the lights"
if "turn off the lights" in text_lower:
    hue_api.turn_off_all()
    self.tts.speak("Lights off.")
    return True
```

---

### Adding a Web Dashboard

A Flask/FastAPI dashboard could show:
- Live conversation transcript
- System status (LLM, STT, Vision)
- Memory contents
- Performance metrics

---

### Voice Personality Tuning

Edit `prompts/system_prompt.txt` to adjust:
- Formality level (more casual / more professional)
- Verbosity (brief responses vs. detailed explanations)
- Expertise areas (add domain-specific instructions)
- Response style (analogies, step-by-step, bullet points in text)

---

## 13. Performance Notes — M1 Mac

| Operation | Typical Latency |
|---|---|
| Wake word detection | < 50 ms |
| Audio recording (VAD stop) | 0 ms overhead (real-time) |
| STT (base, 5s utterance) | 0.4 – 0.8 s |
| LLM first token | 0.5 – 1.5 s |
| LLM full response (100 tokens) | 4 – 7 s |
| TTS synthesis (one sentence) | 0.6 – 1.2 s |
| **Total perceived latency** | **~1.5 – 3 s** |

**Optimization tips for M1 Mac (8 GB):**

- Use `stt.model: tiny` for faster transcription at slight accuracy cost
- Use `llm.model: llama3.2:1b` (1B variant) if RAM is tight — loads in ~1.5 GB
- Quit other memory-intensive apps (Chrome, etc.) when running Sterling
- Keep Ollama warm (don't stop it between sessions) — cold model load takes ~3–5s
- The sentence-streaming TTS approach makes responses feel much faster than they are

---

## 14. Known Limitations

| Limitation | Details |
|---|---|
| TTS requires internet | Edge-TTS streams from Microsoft servers; `say` fallback is offline but lower quality |
| 8 GB RAM is tight | No room for larger models; Llama 3.2 3B is the practical limit |
| No persistent memory (yet) | Memory clears on restart; ChromaDB integration is planned |
| Face enrollment | Drop a photo in vision/faces/ and restart Sterling |
| No real-time interruption | Sterling cannot be interrupted mid-response (planned improvement) |
| STT accuracy | Accents, background noise, and fast speech can reduce accuracy |
| Wake word false positives | Porcupine occasionally triggers on similar-sounding words |
| Single user | Multi-user support (different voices, preferences) is not yet implemented |

---

## 15. Roadmap

### Phase 1 — Core (Current)
- [x] Wake word detection
- [x] Speech-to-text pipeline
- [x] Ollama LLM integration
- [x] Neural TTS (Edge-TTS)
- [x] USB webcam vision (YOLO + face_recognition)
- [x] Session memory
- [x] Sentence-streaming TTS

### Phase 2 — Intelligence
- [ ] ChromaDB long-term memory
- [ ] Project file awareness (read code files, describe them)
- [ ] Custom wake word "Hey Sterling"
- [ ] Interruption handling (stop speaking mid-sentence)
- [ ] Multi-turn clarification ("Did you mean X or Y?")

### Phase 3 — Environment
- [ ] Home automation integration (HomeKit / Home Assistant)
- [ ] Smart display output (show diagrams, code, web search results)
- [ ] Timer and alarm system
- [ ] Calendar / schedule integration

### Phase 4 — Multi-Platform
- [ ] Windows build (GTX 3060 — CUDA acceleration, larger models)
- [ ] Linux build (headless server / systemd service)
- [ ] Cross-platform persistent memory sync

### Phase 5 — Advanced
- [ ] Real-time audio interruption
- [ ] Multi-user voice recognition (identify speaker)
- [ ] Proactive notifications ("Heads up, it's been 2 hours since your last break")
- [ ] Local web search via SearXNG
- [ ] Code execution sandbox ("Sterling, run this script")

---

*Sterling is a continuous improvement project. The architecture is modular by design — swap any
component (STT, LLM, TTS, Vision) independently as better options emerge.*

---
**Sterling v1.0 — M1 Mac Build | May 2026**
