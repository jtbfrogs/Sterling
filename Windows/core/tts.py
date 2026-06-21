"""
Sterling Text-to-Speech  —  Windows Build
==========================================
Primary  : Edge-TTS (Microsoft neural voices, internet required)
Fallback : pyttsx3 (Windows SAPI — fully offline)

Audio playback uses pygame.mixer (replaces macOS afplay).
pygame is cross-platform, handles .mp3 natively, and exposes a
stop() call so interruption works with the same 50 ms polling loop
used on macOS.

Key design
----------
speak_streaming() pipelines synthesis and playback in two threads so
Sterling never goes silent waiting for the next HTTP round-trip:

    LLM tokens ──► sentence buffer ──► SYNTHESIS THREAD ──► audio queue
                                                                   │
                                          PLAYBACK THREAD ◄────────┘
                                          (pygame.mixer, stops on interrupt)

Interruption
------------
Pass a threading.Event as `stop_event`. Set it from any thread to
immediately stop pygame playback and drain the queue.
Sterling's main loop sets this when mic energy is detected mid-speech.
"""

import asyncio
import os
import queue
import re
import subprocess
import tempfile
import threading
from typing import Generator

from utils.logger import setup_logger
from utils.text import clean_for_tts

logger = setup_logger("sterling.tts")

# ── Dependency checks ─────────────────────────────────────────────────────────

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.warning("edge-tts not installed — falling back to pyttsx3.")

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    PYGAME_AVAILABLE = True
except Exception as e:
    PYGAME_AVAILABLE = False
    logger.warning(f"pygame not available ({e}) — audio playback may fail.")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 not installed — offline TTS fallback unavailable.")

# Minimum characters to accumulate before sending a chunk to TTS.
# Too low → many tiny HTTP calls → gaps.  Too high → long wait before first word.
_MIN_CHUNK_CHARS = 80   # default — overridden by tts.stream_chunk_chars in config


class TTS:
    """
    Text-to-speech with streamed LLM output support and interruption.

    Usage::

        tts = TTS()
        tts.speak("Hello, world.")

        stop = threading.Event()
        full = tts.speak_streaming(llm.stream(messages), stop_event=stop)
        # From another thread: stop.set()  →  pygame stops immediately
    """

    def __init__(
        self,
        voice: str = "en-GB-RyanNeural",
        rate: str = "+5%",
        pitch: str = "-5Hz",
        fallback_voice: str = "en-us",      # pyttsx3 voice ID (Windows SAPI)
        stream_chunk_chars: int = _MIN_CHUNK_CHARS,
    ):
        self._voice = voice
        self._rate = rate
        self._pitch = pitch
        self._fallback_voice = fallback_voice
        self._edge_available = EDGE_TTS_AVAILABLE
        self._chunk_chars = stream_chunk_chars

        # Playback state — used by stop() so any thread can interrupt
        self._playing = False
        self._play_lock = threading.Lock()

        # Rolling buffer of what Sterling is currently saying. The wake-word
        # interrupt monitor reads this to reject echoes of Sterling's own voice
        # (Windows has no hardware echo cancellation by default).
        self._spoken_buffer = ""
        self._spoken_lock = threading.Lock()

        logger.info(
            f"TTS initialized — voice: {voice}, "
            f"edge-tts: {'available' if self._edge_available else 'unavailable (using pyttsx3 fallback)'}, "
            f"pygame: {'available' if PYGAME_AVAILABLE else 'UNAVAILABLE'}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def spoken_text(self) -> str:
        """What Sterling is currently saying (for echo suppression)."""
        with self._spoken_lock:
            return self._spoken_buffer

    def _set_spoken(self, text: str):
        with self._spoken_lock:
            self._spoken_buffer = text

    def speak(self, text: str, stop_event: threading.Event = None):
        """
        Synthesize and play text. Blocks until audio finishes (or stop_event is set).
        """
        clean = clean_for_tts(text)
        if not clean.strip():
            return
        self._set_spoken(clean)

        logger.debug(f"Speaking: {clean[:80]}{'...' if len(clean) > 80 else ''}")

        if self._edge_available:
            try:
                self._play_edge(clean, stop_event)
                return
            except Exception as e:
                logger.warning(f"Edge-TTS failed ({e}) — falling back to pyttsx3.")

        self._speak_pyttsx3(clean)

    def speak_streaming(
        self,
        token_generator: Generator[str, None, None],
        stop_event: threading.Event = None,
    ) -> str:
        """
        Consume a streaming LLM token generator, speaking chunks as they arrive.

        Synthesis and playback run in parallel threads (producer/consumer queue)
        so there is no gap between sentences — the next chunk is already
        synthesised before the previous one finishes playing.

        Args:
            token_generator: Generator yielding LLM tokens.
            stop_event:      Optional Event. Set to interrupt immediately.

        Returns:
            The full concatenated response text.
        """
        audio_q: queue.Queue = queue.Queue(maxsize=4)
        full_parts: list[str] = []
        stop = stop_event or threading.Event()
        self._set_spoken("")

        # ── Producer: accumulate text → synthesize → enqueue file paths ──────
        def producer():
            buffer = ""
            try:
                for token in token_generator:
                    if stop.is_set():
                        break
                    buffer += token
                    full_parts.append(token)
                    self._set_spoken("".join(full_parts))

                    sentences = _split_sentences(buffer)
                    # Speak all complete sentences if we have a big enough chunk
                    if len(sentences) > 1:
                        speakable = "".join(sentences[:-1])
                        if len(speakable) >= self._chunk_chars:
                            self._enqueue_synthesis(speakable, audio_q, stop)
                            buffer = sentences[-1]

                # Flush any remaining text
                if buffer.strip() and not stop.is_set():
                    self._enqueue_synthesis(buffer, audio_q, stop)

            except Exception as e:
                logger.error(f"TTS producer error: {e}")
            finally:
                audio_q.put(None)  # sentinel — tells consumer to exit

        # ── Consumer: dequeue file paths → pygame playback ───────────────────
        def consumer():
            while True:
                try:
                    path = audio_q.get(timeout=15)
                except queue.Empty:
                    break
                if path is None:
                    break
                if stop.is_set():
                    _safe_delete(path)
                    continue
                self._play_file(path, stop)

        prod_thread = threading.Thread(target=producer, daemon=True)
        cons_thread = threading.Thread(target=consumer, daemon=True)
        prod_thread.start()
        cons_thread.start()

        # Use timeout-based joins so Ctrl+C / KeyboardInterrupt is always
        # delivered to the main thread even if a child thread is blocking.
        try:
            while prod_thread.is_alive():
                prod_thread.join(timeout=0.05)
            while cons_thread.is_alive():
                cons_thread.join(timeout=0.05)
        except KeyboardInterrupt:
            stop.set()
            self.stop()
            raise

        return "".join(full_parts)

    def stop(self):
        """
        Immediately stop any currently playing audio.
        Call from any thread (e.g. interruption monitor).
        """
        with self._play_lock:
            if self._playing and PYGAME_AVAILABLE:
                try:
                    pygame.mixer.music.stop()
                    logger.debug("TTS interrupted — pygame.mixer stopped.")
                except Exception as e:
                    logger.debug(f"pygame stop error: {e}")
                self._playing = False

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _play_file(self, path: str, stop: threading.Event):
        """
        Load and play an audio file via pygame.mixer.
        Polls every 50 ms so stop_event interrupts within 50 ms.
        """
        if not PYGAME_AVAILABLE:
            logger.warning("pygame unavailable — cannot play audio.")
            _safe_delete(path)
            return
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            with self._play_lock:
                self._playing = True

            # Poll until playback finishes or stop_event fires
            while pygame.mixer.music.get_busy():
                if stop.is_set():
                    pygame.mixer.music.stop()
                    break
                threading.Event().wait(0.05)   # 50 ms poll — same latency as afplay version

            with self._play_lock:
                self._playing = False
        except Exception as e:
            logger.debug(f"pygame playback error: {e}")
            with self._play_lock:
                self._playing = False
        finally:
            # pygame holds a file handle — unload before deleting
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass
            _safe_delete(path)

    def _enqueue_synthesis(self, text: str, q: queue.Queue, stop: threading.Event):
        """Synthesize text to a temp file and put the path in the queue."""
        if stop.is_set():
            return
        clean = clean_for_tts(text)
        if not clean.strip():
            return
        try:
            path = self._synthesize_to_file(clean)
            q.put(path)
        except Exception as e:
            logger.warning(f"Synthesis failed for chunk: {e}")

    def _synthesize_to_file(self, text: str) -> str:
        """Run edge-tts synthesis → temp .mp3 file. Returns the file path."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        asyncio.run(self._async_synthesize(text, path))
        return path

    async def _async_synthesize(self, text: str, path: str):
        communicate = edge_tts.Communicate(
            text=text,
            voice=self._voice,
            rate=self._rate,
            pitch=self._pitch,
        )
        await communicate.save(path)

    def _play_edge(self, text: str, stop_event: threading.Event = None):
        """Synthesize and play a single text block (used by speak())."""
        try:
            path = self._synthesize_to_file(text)
        except Exception as e:
            raise RuntimeError(f"edge-tts synthesis failed: {e}") from e

        stop = stop_event or threading.Event()
        self._play_file(path, stop)

    def _speak_pyttsx3(self, text: str):
        """
        Offline TTS fallback using Windows SAPI via pyttsx3.
        Used when Edge-TTS fails or is unavailable.
        """
        if not PYTTSX3_AVAILABLE:
            logger.error("No TTS available — pyttsx3 not installed. Run: pip install pyttsx3")
            return
        try:
            engine = pyttsx3.init()
            # Optionally set a voice — pyttsx3 uses the Windows default voice
            engine.setProperty("rate", 180)   # words per minute (default ~200)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            logger.error(f"pyttsx3 fallback failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Split on sentence-ending punctuation, but NOT on ellipsis (...).
    The last item may be an incomplete fragment.
    """
    # Temporarily replace ... so it doesn't trigger splits
    text = text.replace("...", "\x00\x00\x00")

    # Split after .  !  ?  followed by whitespace
    parts = re.split(r"(?<=[.!?])\s+", text)

    # Restore ellipsis
    parts = [p.replace("\x00\x00\x00", "...") for p in parts]
    return parts if parts else [text]


def _safe_delete(path: str):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
