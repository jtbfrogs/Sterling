#!/usr/bin/env python3
"""
Sterling - Advanced AI Room Assistant
======================================
Inspired by J.A.R.V.I.S from Iron Man.

Entry point. Loads configuration, initializes all subsystems,
and runs the main wake-word → STT → LLM → TTS loop.

Usage:
    source ster/bin/activate
    python main.py
    python main.py --config path/to/config.yaml
    python main.py --no-vision
"""

import os
import re
import sys
import signal
import argparse
import warnings
from pathlib import Path

# Suppress pkg_resources deprecation spam from face_recognition_models.
# The library still works fine - this is a cosmetic warning from an old dep.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="Deprecated call to `pkg_resources", category=DeprecationWarning)

import pyaudio
import yaml

from core.wake_word import WakeWordDetector
from core.stt import STT
from core.llm import LLM
from core.tts import TTS
from core.memory import Memory
from vision.webcam import WebcamVision
from core.govee import GoveeCloud, GoveeLocal, COLORS
from core.spotify import Spotify
from core.workspace import Workspace, LANGUAGE_ALIASES
from utils.audio import AudioRecorder
from utils.logger import setup_logger
from utils.text import truncate_for_display
from utils.weather import get_weather_context
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap logger (before config is loaded)
# ─────────────────────────────────────────────────────────────────────────────

logger = setup_logger("sterling", level="INFO")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        logger.error(f"Config file not found: {path}")
        logger.error("Copy config.yaml.example to config.yaml and fill in your settings.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_system_prompt(path: str = "prompts/system_prompt.txt") -> str:
    prompt_path = Path(path)
    if not prompt_path.exists():
        logger.warning(f"System prompt not found at {path}. Using minimal default.")
        return "You are Sterling, an advanced AI assistant. Be helpful, clear, and concise."

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ─────────────────────────────────────────────────────────────────────────────
# Sterling
# ─────────────────────────────────────────────────────────────────────────────

class Sterling:
    """
    Main Sterling assistant class.
    Orchestrates all components: wake word → STT → LLM → TTS, with vision side-channel.
    """

    def __init__(self, config: dict, enable_vision: bool = True):
        self._config = config
        self._running = False

        # Reconfigure logger with settings from config
        log_cfg = config.get("logging", {})
        global logger
        logger = setup_logger(
            "sterling",
            level=log_cfg.get("level", "INFO"),
            log_file=log_cfg.get("file") if log_cfg.get("file") else None,
        )

        logger.info("=" * 60)
        logger.info("  STERLING - AI Room Assistant")
        logger.info("=" * 60)
        logger.info("Initializing subsystems...")

        self._init_memory()
        self._init_wake_word()
        self._init_recorder()
        self._init_stt()
        self._init_llm()
        self._init_tts()
        self._init_vision(enable_vision)
        self._init_gesture()
        self._init_govee()
        self._init_spotify()
        self._init_workspace()

        # Gesture wake flag — set by wave callback to trigger an interaction
        import threading as _threading
        self._gesture_wake = _threading.Event()

        logger.info("All subsystems ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────

    def _init_memory(self):
        system_prompt = load_system_prompt()
        mem_cfg = self._config.get("memory", {})
        self._memory = Memory(
            system_prompt=system_prompt,
            max_history=mem_cfg.get("max_history", 10),
            persist=mem_cfg.get("persist", True),
            memory_file=mem_cfg.get("memory_file", "memory.json"),
            recall_turns=mem_cfg.get("recall_turns", 2),
            chroma_enabled=mem_cfg.get("chroma_enabled", True),
            chroma_path=mem_cfg.get("chroma_path", ".chroma"),
            chroma_results=mem_cfg.get("chroma_results", 3),
        )
        logger.info("✓ Memory initialized")

    def _init_wake_word(self):
        ww_cfg = self._config.get("wake_word", {})

        # Phrase list - fallback to audio silence threshold for energy
        audio_cfg = self._config.get("audio", {})
        default_threshold = audio_cfg.get("silence_threshold", 500)

        phrases = ww_cfg.get("phrases", ["hey sterling", "sterling"])
        if isinstance(phrases, str):
            phrases = [phrases]  # handle single-string mistake in config

        self._wake_word = WakeWordDetector(
            phrases=phrases,
            model_size=ww_cfg.get("model_size", "tiny"),
            energy_threshold=ww_cfg.get("energy_threshold", default_threshold),
            silence_duration=ww_cfg.get("silence_duration", 0.6),
            max_buffer_seconds=ww_cfg.get("max_buffer_seconds", 2.5),
        )
        logger.info(
            f"✓ Wake word detector ready - phrases: {phrases}"
        )

    def _init_recorder(self):
        audio_cfg = self._config.get("audio", {})
        self._recorder = AudioRecorder(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            channels=audio_cfg.get("channels", 1),
            chunk_size=audio_cfg.get("chunk_size", 1024),
            silence_threshold=audio_cfg.get("silence_threshold", 500),
            silence_duration=audio_cfg.get("silence_duration", 1.5),
            max_recording_seconds=audio_cfg.get("max_recording_seconds", 30),
        )
        logger.info("✓ Audio recorder initialized")

    def _init_stt(self):
        stt_cfg = self._config.get("stt", {})
        self._stt = STT(
            model_size=stt_cfg.get("model", "base"),
            language=stt_cfg.get("language", "en"),
            device=stt_cfg.get("device", "cpu"),
            compute_type=stt_cfg.get("compute_type", "int8"),
        )
        logger.info("✓ Speech-to-text initialized")

    def _init_llm(self):
        llm_cfg = self._config.get("llm", {})
        self._llm = LLM(
            model=llm_cfg.get("model", "llama3.2"),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            temperature=llm_cfg.get("temperature", 0.7),
            max_tokens=llm_cfg.get("max_tokens", 512),
            context_window=llm_cfg.get("context_window", 4096),
        )
        self._llm_stream = llm_cfg.get("stream", True)
        logger.info("✓ LLM initialized")

    def _init_tts(self):
        tts_cfg = self._config.get("tts", {})
        self._tts = TTS(
            voice=tts_cfg.get("voice", "en-GB-RyanNeural"),
            rate=tts_cfg.get("rate", "+5%"),
            pitch=tts_cfg.get("pitch", "-5Hz"),
            fallback_voice=tts_cfg.get("fallback_voice", "Alex"),
        )
        logger.info("✓ Text-to-speech initialized")

    def _init_vision(self, enable: bool):
        self._vision = None
        vision_cfg = self._config.get("vision", {})

        if not enable or not vision_cfg.get("enabled", False):
            logger.info("  Vision disabled (set vision.enabled: true in config to activate)")
            return

        try:
            cam = WebcamVision(
                device_index      = vision_cfg.get("device_index", 0),
                model_size        = vision_cfg.get("yolo_model", "yolov8n.pt"),
                face_recognition  = vision_cfg.get("face_recognition", True),
                known_faces_dir   = vision_cfg.get("known_faces_dir", "vision/faces"),
                confidence_thresh = vision_cfg.get("confidence_thresh", 0.45),
            )
            cam.start()
            cam.startup_scan()
            self._vision = cam
            logger.info("✓ Vision (webcam + YOLO) initialized")
        except Exception as e:
            logger.warning(f"  Vision unavailable: {e}")
            logger.warning("  Continuing without vision. Check camera is connected.")

    def _init_gesture(self):
        """Initialise gesture detection if enabled in config."""
        self._gesture: "GestureDetector | None" = None
        g_cfg = self._config.get("gestures", {})

        if not g_cfg.get("enabled", False):
            logger.info("  Gesture detection disabled (set gestures.enabled: true to activate)")
            return

        if not self._vision:
            logger.warning("  Gesture detection requires vision — skipping.")
            return

        try:
            from vision.gesture import GestureDetector
            self._gesture = GestureDetector(
                frame_fn       = self._vision.get_frame,
                model_size     = g_cfg.get("model", "yolov8n-pose.pt"),
                poll_interval  = g_cfg.get("poll_interval", 1.0),
                sustain_seconds= g_cfg.get("sustain", 1.5),
                confidence     = g_cfg.get("confidence", 0.45),
            )
            logger.info("✓ Gesture detector initialised (starts at run time)")
        except Exception as e:
            logger.warning(f"  Gesture detection unavailable: {e}")

    def _start_gesture_monitor(self):
        """Register gesture callbacks and start the detector thread."""
        if not self._gesture:
            return

        g_cfg    = self._config.get("gestures", {})
        wave_act = g_cfg.get("wave_action",        "wake")
        hu_act   = g_cfg.get("hands_up_action",    "stop")
        pr_act   = g_cfg.get("point_right_action", "next")
        pl_act   = g_cfg.get("point_left_action",  "prev")

        def _wave():
            logger.info("Gesture: wave — waking Sterling")
            if wave_act == "wake":
                self._gesture_wake.set()

        def _hands_up():
            logger.info("Gesture: hands up — stopping audio")
            if hu_act == "stop":
                self._tts.stop()
                if self._spotify:
                    try:
                        self._spotify.pause()
                    except Exception:
                        pass

        def _point_right():
            logger.info("Gesture: point right — next track")
            if pr_act == "next" and self._spotify:
                try:
                    self._spotify.skip()
                except Exception:
                    pass

        def _point_left():
            logger.info("Gesture: point left — previous track")
            if pl_act == "prev" and self._spotify:
                try:
                    self._spotify.previous()
                except Exception:
                    pass

        self._gesture.on_gesture("wave",        _wave)
        self._gesture.on_gesture("hands_up",    _hands_up)
        self._gesture.on_gesture("point_right", _point_right)
        self._gesture.on_gesture("point_left",  _point_left)
        self._gesture.start()
        logger.info("✓ Gesture monitor active")

    def _init_workspace(self):
        self._workspace: Workspace | None = None
        ws_cfg = self._config.get("workspace", {})
        path   = ws_cfg.get("path", "").strip()

        if not path:
            logger.info("  Workspace disabled (set workspace.path in config to enable)")
            return

        try:
            self._workspace = Workspace(path)
            logger.info(f"✓ Workspace ready - {self._workspace.root}")
        except Exception as e:
            logger.warning(f"  Workspace unavailable: {e}")

    def _init_spotify(self):
        self._spotify: Spotify | None = None
        sp_cfg = self._config.get("spotify", {})

        if not sp_cfg.get("enabled", False):
            logger.info("  Spotify disabled (set spotify.enabled: true in config to activate)")
            return

        client_id     = sp_cfg.get("client_id", "").strip()
        client_secret = sp_cfg.get("client_secret", "").strip()

        if not client_id or not client_secret:
            logger.warning(
                "  Spotify enabled but client_id or client_secret missing in config. "
                "Create an app at developer.spotify.com."
            )
            return

        try:
            self._spotify = Spotify(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=sp_cfg.get("redirect_uri", "http://localhost:8888/callback"),
                cache_path=sp_cfg.get("cache_path", ".spotify_cache"),
            )
            logger.info("✓ Spotify initialised")
        except Exception as e:
            logger.warning(f"  Spotify unavailable: {e}")

    def _init_govee(self):
        self._govee: GoveeCloud | GoveeLocal | None = None
        govee_cfg = self._config.get("govee", {})

        if not govee_cfg.get("enabled", False):
            logger.info("  Govee disabled (set govee.enabled: true in config to activate)")
            return

        devices = govee_cfg.get("devices", [])
        if not devices:
            logger.warning(
                "  Govee enabled but no devices configured. "
                "Run: python scripts/discover_govee.py"
            )
            return

        try:
            api_key = govee_cfg.get("api_key", "").strip()
            if api_key:
                self._govee = GoveeCloud(api_key=api_key, devices=devices)
                logger.info(f"✓ Govee (cloud API) - {len(devices)} device(s)")
            else:
                self._govee = GoveeLocal(
                    devices=devices,
                    discovery_timeout=govee_cfg.get("discovery_timeout", 3.0),
                )
                logger.info(f"✓ Govee (local LAN) - {len(devices)} device(s)")
        except Exception as e:
            logger.warning(f"  Govee unavailable: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        """Start Sterling and block until shutdown."""
        self._running = True

        startup_msg = self._config.get("sterling", {}).get(
            "startup_message", "Sterling online. How may I assist you?"
        )
        self._tts.speak(startup_msg)

        # Enable shared audio stream: one persistent CoreAudio input for the
        # entire session.  Wake-word detection and recording take turns reading
        # from it sequentially — no open/close, no two-stream conflict (err=-50).
        self._recorder.enable_shared_mode()
        self._wake_word.start(external_read=self._recorder.read_raw)

        # Start optional background monitors
        self._start_gesture_monitor()
        self._start_presence_monitor()

        logger.info("Listening for wake word...")
        logger.info("-" * 60)

        # Run the blocking wake-word loop in a daemon thread.
        # PyAudio's stream.read() is a blocking C call - on macOS it swallows
        # Ctrl+C because Python only delivers signals between bytecode instructions,
        # not during C extension calls. Keeping the main thread in a Python-level
        # join(timeout) loop means Ctrl+C always gets through.
        import threading
        _loop = threading.Thread(target=self._run_loop, daemon=True)
        _loop.start()

        try:
            while self._running and _loop.is_alive():
                _loop.join(timeout=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _start_presence_monitor(self):
        """Optional background thread — detects when someone enters/leaves."""
        p_cfg = self._config.get("presence", {})
        if not p_cfg.get("enabled", False) or not self._vision:
            return

        import threading, time as _time
        interval     = p_cfg.get("check_interval", 30)
        greet        = p_cfg.get("greet_on_enter", False)
        lights_on    = p_cfg.get("lights_on_enter", False)
        lights_off   = p_cfg.get("lights_off_on_leave", False)
        leave_delay  = p_cfg.get("leave_delay", 300)

        def _monitor():
            was_present = False
            while self._running:
                _time.sleep(interval)
                try:
                    is_present = self._vision.check_presence()
                except Exception:
                    continue

                if is_present and not was_present:
                    logger.info("Presence: person detected — room occupied.")
                    if lights_on and self._govee and self._govee.has_devices:
                        try:
                            self._govee.turn_on()
                        except Exception:
                            pass
                    if greet and self._running:
                        self._tts.speak("Welcome back.")

                elif not is_present and was_present:
                    logger.info("Presence: room appears empty.")
                    if lights_off and self._govee and self._govee.has_devices:
                        _time.sleep(leave_delay)
                        if not self._vision.check_presence():
                            try:
                                self._govee.turn_off()
                                logger.info("Presence: lights off (room still empty).")
                            except Exception:
                                pass

                was_present = is_present

        threading.Thread(target=_monitor, daemon=True, name="sterling-presence").start()
        logger.info(f"✓ Presence monitor active (every {interval}s)")

    def _run_loop(self):
        """Wake-word → interaction loop. Runs in a daemon thread."""
        try:
            while self._running:
                # Gesture wave can trigger an interaction without a wake phrase
                if self._gesture_wake.is_set():
                    self._gesture_wake.clear()
                    self._handle_interaction()
                    continue
                if self._wake_word.listen():
                    self._handle_interaction()
        except Exception as e:
            logger.error(f"Run loop error: {e}")
            self._running = False

    # ─────────────────────────────────────────────────────────────────────────
    # Interaction handling
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_interaction(self):
        """
        Called once when the wake word fires. Pauses the wake word stream,
        acknowledges, then hands off to _conversation_loop() which keeps
        listening until the user goes quiet. Resumes the wake word stream
        when done.
        """
        self._wake_word.pause()
        try:
            self._tts.speak("Yes?")
            self._conversation_loop()
        finally:
            self._wake_word.resume()
            logger.info("Listening for wake word...")
            logger.info("-" * 60)

    def _conversation_loop(self):
        """
        Natural multi-turn conversation after the wake word fires.
        No wake word needed between turns - Sterling listens continuously
        and responds until the user goes silent for `conversation.timeout`
        seconds, or says a shutdown/goodbye phrase.

        Stream discipline: the wake word stream is already PAUSED when this
        runs. listen_for_speech() opens and fully closes its own stream so
        CoreAudio never sees two simultaneous input streams.
        """
        conv_cfg  = self._config.get("conversation", {})
        timeout   = conv_cfg.get("timeout", 20)   # seconds of silence before sleeping

        logger.info(f"Conversation mode active - {timeout}s idle timeout")

        # One stream open for the entire conversation - eliminates the
        # ~200 ms open/close gap that was clipping the first word of each turn.
        self._recorder.open_stream()
        try:
          while self._running:
            # Wait for the user to speak (or time out)
            logger.debug("Waiting for follow-up...")
            audio = self._recorder.listen_for_speech(timeout_seconds=timeout)

            if audio is None:
                # Check if this is a shutdown-triggered abort rather than a real timeout
                if not self._running:
                    break
                # Nobody spoke within the timeout window
                sleep_msg = conv_cfg.get(
                    "sleep_message", "Going quiet. Say my name when you need me."
                )
                self._tts.speak(sleep_msg)
                logger.info(f"Conversation idle for {timeout}s - returning to wake word mode.")
                break

            # Transcribe what the user said
            text = self._stt.transcribe(audio)
            if not text:
                logger.debug("No speech content - continuing to listen.")
                continue

            logger.info(f"User said: {truncate_for_display(text, 100)}")

            # Built-in commands (shutdown, clear memory, etc.)
            if self._handle_command(text):
                if not self._running:
                    break       # actual shutdown — leave conversation
                continue        # vision / diagnostics / memory clear — already spoke, keep listening

            # Wind-down detection - let LLM say goodbye first, then return to wake word
            winding_down = self._is_winddown(text)

            # Track which intent fired so later parsers can skip
            intent_handled = False

            # Face enrollment — "remember this person as James"
            enroll_name = self._parse_enroll_intent(text)
            if enroll_name and self._vision:
                result = self._vision.enroll_face(enroll_name)
                llm_inject = f"{text}\n\n[Face enrollment result: {result}]"
                self._memory.add_user(llm_inject)
                logger.info(f"Face enrollment: {result}")
                full_response = ""
                if self._llm_stream:
                    full_response, was_interrupted = self._speak_interruptible(
                        self._tts.speak_streaming,
                        self._llm.stream(self._memory.get_messages()),
                    )
                if full_response:
                    self._memory.add_assistant(full_response)
                intent_handled = True
                if winding_down:
                    break
                continue

            # Object tracking — "where's my phone / have you seen my keys"
            if not intent_handled and self._vision and self._is_tracking_query(text):
                result = self._vision._tracker.find(text)
                if result:
                    llm_inject = f"{text}\n\n[Object tracking: {result}]"
                else:
                    llm_inject = f"{text}\n\n[Object tracking: I haven't seen that item yet. It will be tracked next time it's visible on camera.]"
                text = llm_inject   # feed into LLM below
                intent_handled = True

            # Spotify control - execute immediately, let LLM respond naturally
            spotify_context = None
            if self._spotify:
                spotify_action = self._parse_spotify_intent(text)
                if spotify_action:
                    self._execute_spotify_command(spotify_action)
                    if spotify_action["action"] == "now_playing":
                        spotify_context = spotify_action.get("result")
                    intent_handled = True

            # Light control - execute immediately, then let LLM respond naturally
            if not intent_handled and self._govee and self._govee.has_devices:
                light_action = self._parse_light_intent(text)
                if light_action:
                    self._execute_light_command(light_action)
                    intent_handled = True

            # Project creation - only if no other intent already fired
            if not intent_handled and self._workspace:
                project_intent = self._parse_project_intent(text)
                if project_intent:
                    project_result = self._create_project(project_intent)
                    if project_result:
                        llm_text = f"{text}\n\n[{project_result}]"

            llm_text = text

            # Weather — inject live conditions so LLM answers from real data
            if self._is_weather_query(text):
                location = (
                    self._extract_weather_location(text)
                    or self._config.get("weather", {}).get("location", "")
                )
                if location:
                    conditions = get_weather_context(location)
                    if conditions:
                        llm_text = (
                            f"{text}\n\n"
                            f"[Use only the following data to answer — do not guess: "
                            f"{conditions}]"
                        )
                        logger.info(f"Weather context injected for: {location}")

            # Vision — inject live scene description so LLM can answer
            # naturally: “what am I holding”, “what do you see”, etc.
            if not intent_handled and self._is_vision_query(text):
                if not self._vision:
                    llm_text = f"{text}\n\n[Camera is offline — you cannot see anything.]"
                else:
                    scene = self._vision.get_scene_description()
                    llm_text = (
                        f"{text}\n\n"
                        f"[Camera feed — use this to answer, do not guess: {scene}]"
                    )
                    logger.info(f"Vision context injected: {scene[:80]}")
                intent_handled = True

            # Time / date — inject so LLM can answer naturally
            if self._is_time_query(text):
                now = datetime.now()
                time_str = now.strftime("%I:%M %p").lstrip("0")  # e.g. "3:45 PM"
                date_str = now.strftime("%A, %B %d, %Y")         # e.g. "Monday, June 9 2025"
                llm_text = f"{text}\n\n[Current local time: {time_str}. Today is {date_str}.]"
                logger.info(f"Time context injected: {time_str}, {date_str}")

            if spotify_context:
                llm_text = f"{text}\n\n[Currently playing: {spotify_context}]"

            # LLM response
            self._memory.add_user(llm_text)
            logger.info("Generating response...")

            full_response = ""

            if self._llm_stream:
                full_response, was_interrupted = self._speak_interruptible(
                    self._tts.speak_streaming,
                    self._llm.stream(self._memory.get_messages()),
                )
            else:
                full_response = self._llm.chat(self._memory.get_messages())
                _, was_interrupted = self._speak_interruptible(
                    self._tts.speak, full_response
                )

            if was_interrupted:
                logger.info("Wake word interrupt - discarding partial response, listening again.")
                continue

            if full_response:
                self._memory.add_assistant(full_response)
                logger.info(f"Sterling: {truncate_for_display(full_response, 100)}")

            logger.info(
                f"[Session: {self._memory.session_duration}, "
                f"{self._memory.turn_count} turns]"
            )

            # TTS has finished - safe to return to wake word now
            if winding_down:
                logger.info("Conversation wound down - returning to wake word.")
                break
        finally:
            self._recorder.close_stream()

    def _handle_command(self, text: str) -> bool:
        """
        Check for special voice commands that bypass the LLM.

        Returns:
            True if the command was handled (skip LLM), False otherwise.
        """
        t = text.lower().strip()

        # ── Shutdown ──────────────────────────────────────────────────────────
        if any(phrase in t for phrase in [
            "shut down", "shutdown", "power off", "go offline"
        ]):
            shutdown_msg = self._config.get("sterling", {}).get(
                "shutdown_message", "Understood. Powering down. Goodbye."
            )
            self._tts.speak(shutdown_msg)
            self._running = False
            return True

        # ── Memory management ─────────────────────────────────────────────────
        if any(phrase in t for phrase in [
            "clear memory", "clear your memory", "forget everything",
            "reset conversation", "start fresh"
        ]):
            self._memory.clear()
            self._tts.speak(
                "Memory cleared. I've forgotten our previous conversation. Starting fresh."
            )
            return True

        # ── Session status ────────────────────────────────────────────────────
        if any(phrase in t for phrase in [
            "how long have we been talking", "session duration",
            "how long have you been running"
        ]):
            self._tts.speak(
                f"We've been talking for {self._memory.session_duration}, "
                f"across {self._memory.turn_count} exchanges."
            )
            return True

        # Vision queries are handled via LLM context injection
        # (see _is_vision_query() + _conversation_loop).
        # diagnostics
        if any(phrase in t for phrase in [
            "run diagnostics", "system status", "what's working",
            "what is working", "systems check", "your status",
        ]):
            self._speak_diagnostics()
            return True

        return False

    def _speak_diagnostics(self):
        """Speak a plain-English summary of which systems are actually online."""
        parts = []
        parts.append("camera is " + ("online" if self._vision else "offline"))

        if self._govee and self._govee.has_devices:
            n = len(self._govee.device_names)
            parts.append(f"lights are online with {n} device{'s' if n != 1 else ''} connected")
        else:
            parts.append("lights are offline or not set up")

        parts.append("Spotify is " + ("online" if self._spotify else "offline or not configured"))
        parts.append("weather is available")
        parts.append("workspace is " + ("ready" if self._workspace else "not configured"))

        joined = ", ".join(parts[:-1]) + ", and " + parts[-1]
        self._tts.speak(f"Here's where things stand. {joined.capitalize()}.")
        logger.info("Diagnostics spoken.")

    # ─────────────────────────────────────────────────────────────────────────
    # Vision helpers
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # Wake word interruption
    # ─────────────────────────────────────────────────────────────────────────

    def _speak_interruptible(self, speak_func, *args) -> tuple[str, bool]:
        """
        Run a TTS call with wake word interruption support.

        Swaps the microphone from the recorder to the wake word detector
        for the duration of playback - the only safe way to run both
        wake word detection and recording without CoreAudio conflicts.

        A background thread calls wake_word.listen() in a loop. If the
        wake phrase is detected, stop_event is set and TTS kills afplay
        within its 50 ms poll cycle.

        Returns:
            (result, was_interrupted)
            result         - return value of speak_func (str for speak_streaming, None for speak)
            was_interrupted - True if wake word cut the speech short
        """
        import threading

        stop_event  = threading.Event()
        interrupted = threading.Event()

        # Hand the microphone to the wake word detector
        self._recorder.close_stream()
        self._wake_word.resume()

        def _monitor():
            while not stop_event.is_set():
                try:
                    # Pass stop_event so listen() skips Whisper transcription
                    # (~200 ms) once we've signalled stop - exits in ~32 ms instead.
                    if self._wake_word.listen(stop_event=stop_event):
                        logger.info("Wake word detected mid-speech - interrupting.")
                        interrupted.set()
                        stop_event.set()
                except Exception as e:
                    logger.debug(f"Interrupt monitor error: {e}")
                    break

        monitor = threading.Thread(target=_monitor, daemon=True)
        monitor.start()

        try:
            result = speak_func(*args, stop_event=stop_event)
        finally:
            stop_event.set()              # tell monitor: skip transcription, exit loop
            monitor.join(timeout=0.5)     # wait ~32 ms (listen() skips Whisper when stop set)
            self._wake_word.pause()       # ONLY close wake stream after monitor has exited
            self._recorder.open_stream()  # one input stream at a time — always safe here

        return result, interrupted.is_set()

    # ─────────────────────────────────────────────────────────────────────────
    # Conversation wind-down
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_spotify_intent(self, text: str) -> dict | None:
        """
        Parse user speech for Spotify control intent.
        Returns an action dict or None.

        Action shapes:
            {"action": "play",        "query": "Radiohead"}  # search + play
            {"action": "play",        "query": None}          # resume
            {"action": "pause"}
            {"action": "skip"}
            {"action": "previous"}
            {"action": "volume_up"}
            {"action": "volume_down"}
            {"action": "volume_set",  "value": 50}
            {"action": "now_playing"}
        """
        t = text.lower().strip()

        # What's playing
        if any(p in t for p in ("what's playing", "what is playing", "what song", "who is this", "who's this")):
            return {"action": "now_playing"}

        # Pause / stop music
        if any(p in t for p in ("pause", "stop the music", "stop music", "mute spotify")):
            if "spotify" in t or "music" in t or "song" in t or t in ("pause", "stop"):
                return {"action": "pause"}

        # Resume
        if any(p in t for p in ("resume", "unpause", "continue playing", "continue the music")):
            return {"action": "play", "query": None}

        # Skip / next
        if any(p in t for p in ("skip", "next song", "next track", "skip this")):
            return {"action": "skip"}

        # Previous
        if any(p in t for p in ("previous song", "previous track", "go back", "last song")):
            return {"action": "previous"}

        # Volume set - "set volume to 50" / "volume at 40 percent"
        vol_match = re.search(r'volume.*?(\d+)|set.*?volume.*(\d+)|(\d+).*?volume', t)
        if vol_match:
            val = next(v for v in vol_match.groups() if v is not None)
            return {"action": "volume_set", "value": int(val)}

        # Volume up / down
        if any(p in t for p in ("volume up", "louder", "turn it up", "turn up the music")):
            return {"action": "volume_up"}
        if any(p in t for p in ("volume down", "quieter", "turn it down", "turn down the music")):
            return {"action": "volume_down"}

        # Play [query] - extract what comes after "play"
        if "play" in t:
            # Strip filler so "play some music" doesn't search for "some music"
            filler = ("some music", "something", "music", "a song", "anything", "spotify")
            after  = re.split(r'\bplay\b', t, maxsplit=1)[-1].strip()
            if not after or after in filler:
                return {"action": "play", "query": None}
            # Clean up common prefixes
            for f in ("me ", "some ", "a bit of "):
                if after.startswith(f):
                    after = after[len(f):]
            return {"action": "play", "query": after.strip()}

        return None

    def _execute_spotify_command(self, action: dict):
        """Execute a parsed Spotify action."""
        kind = action["action"]
        try:
            if kind == "play":
                self._spotify.play(action.get("query"))
            elif kind == "pause":
                self._spotify.pause()
            elif kind == "skip":
                self._spotify.skip()
            elif kind == "previous":
                self._spotify.previous()
            elif kind == "volume_up":
                self._spotify.volume_up()
            elif kind == "volume_down":
                self._spotify.volume_down()
            elif kind == "volume_set":
                self._spotify.set_volume(action["value"])
            elif kind == "now_playing":
                track = self._spotify.now_playing()
                # Inject into LLM context is handled in conversation loop
                action["result"] = track  # stored for context injection below
            logger.info(f"Spotify command executed: {action}")
        except Exception as e:
            logger.error(f"Spotify command failed: {e}")

    def _extract_weather_location(self, text: str) -> str:
        """
        Try to pull a location out of a weather question.
        e.g. "what's the weather in Denver" → "Denver"
             "weather in New York tomorrow" → "New York"
        Returns empty string if no location found (caller falls back to config default).
        """
        match = re.search(
            r'(?:weather|temperature|forecast|raining|snowing|hot|cold|warm)'
            r'.{0,20}\bin\s+([A-Za-z][A-Za-z\s,]{1,40}?)'
            r'(?:\s*\?|\s*$|\s+tomorrow|\s+today|\s+this)',
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return ""

    # ─────────────────────────────────────────────────────────────────────────
    # Project creation
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_project_intent(self, text: str) -> dict | None:
        """
        Detect a project creation request and extract language, name, description.

        Returns a dict or None:
            {"language": "python", "name": "tracker", "description": "scrapes weather data"}
        """
        t = text.lower()

        # Require an explicit create verb alongside an explicit project noun.
        # Kept strict to avoid false positives from Spotify/weather/light phrases.
        create_words  = ("create", "make me a", "make a", "build", "set up", "setup", "initialise", "initialize")
        project_words = ("project", "application", "script", "program")

        if not any(w in t for w in create_words):
            return None
        if not any(w in t for w in project_words):
            return None

        # Detect language - default to Python
        language = "python"
        for alias, lang in LANGUAGE_ALIASES.items():
            if alias in t:
                language = lang
                break

        # Extract project name after "called" or "named"
        name_match = re.search(
            r'(?:called|named)\s+([a-zA-Z0-9][a-zA-Z0-9_\s-]*?)'
            r'(?:\s+that|\s+which|\s+to\s|\s*\?|\s*$)',
            t,
        )
        if not name_match:
            return None

        name = name_match.group(1).strip()

        # Extract optional description after "that" / "which" / "to"
        desc_match = re.search(r'(?:that|which)\s+(.+?)(?:\?|$)', t)
        description = desc_match.group(1).strip() if desc_match else ""

        return {"language": language, "name": name, "description": description}

    def _create_project(self, intent: dict) -> str:
        """
        Scaffold the project and optionally generate starter code via LLM.
        Returns a plain-English summary injected into the LLM context.
        """
        language    = intent["language"]
        name        = intent["name"]
        description = intent.get("description", "")
        code        = None

        # Generate starter code if a description was given
        if description:
            logger.info(f"Generating {language} code for: {description}")
            code = self._generate_code(language, description)

        try:
            result = self._workspace.create_project(
                name=name,
                language=language,
                code=code,
                description=description,
            )
        except Exception as e:
            logger.error(f"Project creation failed: {e}")
            return f"Project creation failed: {e}"

        venv_note = " A venv was set up inside the folder." if language == "python" else ""
        code_note = f" Generated starter code based on: {description}." if code else " A blank template was used."
        return (
            f"{language.capitalize()} project '{name}' created at {result['path']}."
            f"{code_note}{venv_note}"
        )

    def _generate_code(self, language: str, description: str) -> str:
        """
        Ask the LLM to write code for the given language and description.
        Uses a dedicated code-generation prompt - overrides the normal system prompt.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert {language} programmer. "
                    f"Output ONLY clean, working {language} code with no explanation, "
                    f"no markdown fences, and no commentary. "
                    f"The code should be well-structured and follow best practices."
                ),
            },
            {
                "role": "user",
                "content": f"Write a {language} program that {description}.",
            },
        ]
        try:
            # Use a higher token limit for code generation
            original_max = self._llm._max_tokens
            self._llm._max_tokens = 1024
            code = self._llm.chat(messages)
            self._llm._max_tokens = original_max
            return code.strip()
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return ""

    def _parse_enroll_intent(self, text: str) -> str | None:
        """
        Detect a voice request to enrol a new face.
        Returns the name to save, or None.

        Triggers on phrases like:
            "remember this person as James"
            "save this face as Sarah"
            "enroll this person as Dad"
        """
        t = text.lower().strip()
        patterns = [
            r"remember (?:this person|this face|them|him|her) as ([\w\s]+)",
            r"save (?:this person|this face|them|him|her) as ([\w\s]+)",
            r"enrol(?:l)? (?:this person|this face|them|him|her) as ([\w\s]+)",
            r"(?:call|name) (?:this person|them|him|her) ([\w\s]+)",
            r"(?:that'?s?|this is) ([\w]+)(?: right)?",   # "that's James" — short form
        ]
        for pattern in patterns:
            m = re.search(pattern, t)
            if m:
                name = m.group(1).strip().rstrip(".!?").strip()
                if 1 < len(name) < 30 and name not in ("me", "you", "us", "him", "her"):
                    return name
        return None

    def _is_tracking_query(self, text: str) -> bool:
        """Returns True if the user is asking where an object was last seen."""
        t = text.lower()
        return any(phrase in t for phrase in [
            "where's my", "where is my", "where did i put",
            "have you seen my", "seen my", "find my",
            "where did i leave", "where are my",
        ])

    def _is_vision_query(self, text: str) -> bool:
        """
        Returns True if the user is asking about what the camera sees.
        Covers broad natural phrasing: objects, people, actions, held items.
        """
        t = text.lower()
        return any(phrase in t for phrase in [
            # What do you see
            "what do you see", "what can you see", "what are you seeing",
            "what's in front of you", "what is in front of you",
            "look at this", "what is this", "what are you looking at",
            "describe what you see", "describe the room", "describe the scene",
            # Who / people
            "who's in the room", "who is in the room", "who do you see",
            "who's there", "who is there", "anyone in the room",
            "can you see me", "do you see me",
            # Holding / hands
            "what am i holding", "what's in my hand", "what is in my hand",
            "what am i carrying", "what do i have in my hand",
            "what's that in my hand", "what is that",
            # Actions / context
            "what am i doing", "what's going on", "what do you notice",
            "anything interesting", "what's around",
        ])

    def _is_time_query(self, text: str) -> bool:
        """Returns True if the user is asking for the current time or date."""
        t = text.lower()
        return any(phrase in t for phrase in [
            "what time", "what's the time", "do you know the time",
            "what's today", "what day is it", "what's the date",
            "today's date", "what is today", "day of the week",
            "what year", "what month",
        ])

    def _is_weather_query(self, text: str) -> bool:
        """Returns True if the user is asking about the weather."""
        t = text.lower()
        return any(phrase in t for phrase in [
            "weather",
            "temperature",
            "how hot",
            "how cold",
            "how warm",
            "raining",
            "snowing",
            "outside",
            "bring a jacket",
            "what's it like",
            "forecast",
        ])

    def _is_winddown(self, text: str) -> bool:
        """
        Returns True if the user is wrapping up the conversation.
        Sterling will still respond via LLM, then return to wake word mode
        once TTS has finished - so Sterling always gets to say goodbye first.
        """
        t = text.lower()
        return any(phrase in t for phrase in [
            "i'm done",
            "im done",
            "goodbye",
            "talk later",
        ])

    # ─────────────────────────────────────────────────────────────────────────
    # Light control
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_light_intent(self, text: str) -> dict | None:
        """
        Parse user speech for light control intent.
        Returns an action dict or None if no light command detected.

        Action shapes:
            {"action": "on"}
            {"action": "off"}
            {"action": "color",      "color": "blue"}
            {"action": "brightness", "value": 50}
        """
        t = text.lower()

        # Words that suggest the user is talking about lights
        light_words = ("light", "lights", "lamp", "lamps", "bulb", "bulbs", "led", "strip")
        has_light = any(w in t for w in light_words)

        # ── On / Off ─────────────────────────────────────────────────────────
        if has_light or any(p in t for p in ("turn on", "turn off", "switch on", "switch off")):
            # Check off before on to avoid "turn on" matching the "on" in "turn off"
            if "off" in t:
                return {"action": "off"}
            if "on" in t:
                return {"action": "on"}

        # ── Color ─────────────────────────────────────────────────────────────
        color_triggers = ("color", "colour", "make", "set", "change", "switch")
        if has_light or any(w in t for w in color_triggers):
            for color_name in COLORS:
                if color_name in t:
                    return {"action": "color", "color": color_name}

        # ── Brightness ────────────────────────────────────────────────────────
        bright_words = ("dim", "dimmer", "darker", "bright", "brighter", "brightness", "percent", "%")
        if has_light or any(w in t for w in bright_words):
            # Explicit percentage: "50%" or "50 percent"
            match = re.search(r'(\d+)\s*(?:%|percent)', t)
            if match:
                return {"action": "brightness", "value": int(match.group(1))}
            if any(w in t for w in ("dim", "dimmer", "darker")):
                return {"action": "brightness", "value": 20}
            if any(w in t for w in ("full", "max", "maximum")):
                return {"action": "brightness", "value": 100}
            if any(w in t for w in ("bright", "brighter")):
                return {"action": "brightness", "value": 80}

        return None

    def _execute_light_command(self, action: dict):
        """Execute a parsed light action against the Govee controller."""
        try:
            kind = action["action"]
            if kind == "on":
                self._govee.turn_on()
            elif kind == "off":
                self._govee.turn_off()
            elif kind == "color":
                self._govee.set_color_by_name(action["color"])
            elif kind == "brightness":
                self._govee.set_brightness(action["value"])
            logger.info(f"Light command executed: {action}")
        except Exception as e:
            logger.error(f"Light command failed: {e}")

    def _report_faces(self):
        """Report recognised faces from the webcam."""
        try:
            blocks = self._vision.get_learned_blocks()
            if not blocks:
                self._tts.speak("I don't see anyone I recognise right now.")
                return

            names = [b.label for b in blocks if b.label not in ("person", "unknown", "")]
            if not names:
                self._tts.speak("There's someone in frame but I don't recognise them.")
                return

            if len(names) == 1:
                self._tts.speak(f"I can see {names[0]}.")
            else:
                self._tts.speak(f"I can see {', '.join(names[:-1])} and {names[-1]}.")

        except Exception as e:
            logger.error(f"Vision query failed: {e}")
            self._tts.speak("Vision system ran into an error.")

    def _report_objects(self):
        """Report all detected objects from the webcam."""
        try:
            blocks, _ = self._vision.get_all()
            if not blocks:
                self._tts.speak("Nothing in frame at the moment.")
                return

            from collections import Counter
            counts = Counter(b.label for b in blocks)
            parts  = []
            for label, count in counts.most_common():
                if count == 1:
                    parts.append(f"a {label}")
                else:
                    plural = label + "s" if not label.endswith("s") else label
                    parts.append(f"{count} {plural}")

            if len(parts) == 1:
                self._tts.speak(f"I can see {parts[0]}.")
            else:
                self._tts.speak(f"I can see {', '.join(parts[:-1])} and {parts[-1]}.")

        except Exception as e:
            logger.error(f"Vision query failed: {e}")
            self._tts.speak("Vision system ran into an error.")

    # ─────────────────────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────────────────────

    def shutdown(self):
        """Gracefully release all resources."""
        logger.info("Shutting down Sterling...")
        self._running = False

        try:
            self._memory.end_session()
        except Exception:
            pass

        try:
            self._wake_word.stop()
        except Exception:
            pass

        try:
            self._recorder.terminate()
        except Exception:
            pass

        try:
            if self._vision:
                self._vision.disconnect()
        except Exception:
            pass

        try:
            if self._govee:
                self._govee.close()
        except Exception:
            pass

        logger.info("Sterling offline.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sterling - AI Room Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with default config.yaml
  python main.py --config dev.yaml        # Use a different config file
  python main.py --no-vision              # Disable webcam vision
        """,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--no-vision", action="store_true",
        help="Disable vision regardless of config"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    sterling = Sterling(
        config=config,
        enable_vision=not args.no_vision,
    )

    # Backup signal handler - the run() loop catches KeyboardInterrupt itself,
    # but this fires if the signal lands outside the join() window.
    def handle_sigint(signum, frame):
        print()
        # Save memory then force-exit immediately.
        # PyAudio's Pa_StopStream can deadlock on macOS if we wait for
        # the graceful shutdown path — os._exit bypasses that entirely.
        try:
            sterling._memory.end_session()
        except Exception:
            pass
        os._exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    sterling.run()


if __name__ == "__main__":
    main()
