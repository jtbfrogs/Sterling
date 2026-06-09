"""
Sterling Text-to-Speech
=======================
Primary  : Edge-TTS (Microsoft neural voices)
Fallback : macOS `say` command (offline)

Key design
----------
speak_streaming() pipelines synthesis and playback in two threads so
Sterling never goes silent waiting for the next HTTP round-trip:

    LLM tokens ──► sentence buffer ──► SYNTHESIS THREAD ──► audio queue
                                                                   │
                                             PLAYBACK THREAD ◄─────┘
                                             (afplay, kills on interrupt)

Interruption
------------
Pass a threading.Event as `stop_event`. Set it from any thread to
immediately kill the current afplay process and drain the queue.
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

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.warning("edge-tts not installed — falling back to macOS `say`.")

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
        # From another thread: stop.set()  →  afplay is killed immediately
    """

    def __init__(
        self,
        voice: str = "en-GB-RyanNeural",
        rate: str = "+5%",
        pitch: str = "-5Hz",
        fallback_voice: str = "Alex",
        stream_chunk_chars: int = _MIN_CHUNK_CHARS,
    ):
        self._voice = voice
        self._rate = rate
        self._pitch = pitch
        self._fallback_voice = fallback_voice
        self._edge_available = EDGE_TTS_AVAILABLE
        self._chunk_chars = stream_chunk_chars

        # The current afplay Popen object — stored so it can be killed on interrupt
        self._current_proc: subprocess.Popen = None
        self._proc_lock = threading.Lock()

        logger.info(
            f"TTS initialized — voice: {voice}, "
            f"edge-tts: {'available' if self._edge_available else 'unavailable (using say fallback)'}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def speak(self, text: str, stop_event: threading.Event = None):
        """
        Synthesize and play text. Blocks until audio finishes (or stop_event is set).
        """
        clean = clean_for_tts(text)
        if not clean.strip():
            return

        logger.debug(f"Speaking: {clean[:80]}{'...' if len(clean) > 80 else ''}")

        if self._edge_available:
            try:
                self._play_edge(clean, stop_event)
                return
            except Exception as e:
                logger.warning(f"Edge-TTS failed ({e}) — falling back to say.")

        self._speak_say(clean)

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

        # ── Producer: accumulate text → synthesize → enqueue file paths ──────
        def producer():
            buffer = ""
            try:
                for token in token_generator:
                    if stop.is_set():
                        break
                    buffer += token
                    full_parts.append(token)

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

        # ── Consumer: dequeue file paths → afplay ────────────────────────────
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
                try:
                    proc = subprocess.Popen(
                        ["afplay", path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    with self._proc_lock:
                        self._current_proc = proc

                    # Poll so we can react to stop_event quickly
                    while proc.poll() is None:
                        if stop.is_set():
                            proc.kill()
                            break
                        threading.Event().wait(0.05)  # 50 ms poll

                    with self._proc_lock:
                        self._current_proc = None
                except Exception as e:
                    logger.debug(f"afplay error: {e}")
                finally:
                    _safe_delete(path)

        prod_thread = threading.Thread(target=producer, daemon=True)
        cons_thread = threading.Thread(target=consumer, daemon=True)
        prod_thread.start()
        cons_thread.start()

        # Use timeout-based joins so Ctrl+C / KeyboardInterrupt is always
        # delivered to the main thread even if a child thread is blocking.
        # 50 ms poll keeps latency tight while still being interruptible.
        try:
            while prod_thread.is_alive():
                prod_thread.join(timeout=0.05)
            while cons_thread.is_alive():
                cons_thread.join(timeout=0.05)
        except KeyboardInterrupt:
            stop.set()          # signal both threads to exit
            self.stop()         # kill afplay immediately
            raise               # re-raise so Sterling can shut down cleanly

        return "".join(full_parts)

    def stop(self):
        """
        Immediately kill any currently playing audio.
        Call from any thread (e.g. interruption monitor).
        """
        with self._proc_lock:
            if self._current_proc and self._current_proc.poll() is None:
                self._current_proc.kill()
                logger.debug("TTS interrupted — afplay killed.")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

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
        try:
            proc = subprocess.Popen(
                ["afplay", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._proc_lock:
                self._current_proc = proc
            while proc.poll() is None:
                if stop.is_set():
                    proc.kill()
                    break
                threading.Event().wait(0.05)
            with self._proc_lock:
                self._current_proc = None
        finally:
            _safe_delete(path)

    def _speak_say(self, text: str):
        subprocess.run(["say", "-v", self._fallback_voice, text], check=False)


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
