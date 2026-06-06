"""
Sterling Wake Word Detector — Whisper-based
============================================
Uses faster-whisper (tiny model) with energy VAD to detect any spoken
wake phrase. No training required, no API key, fully offline.

How it works
------------
1.  A PyAudio stream is read in small 32 ms chunks.
2.  RMS energy is checked each chunk (lightweight voice-activity detection).
3.  When energy crosses the threshold, audio is accumulated into a buffer.
4.  After sustained silence (or the buffer fills), the clip is transcribed
    with the faster-whisper tiny model (< 200 ms on M1).
5.  If any configured wake phrase appears in the transcript, returns True.

Advantages over keyword-spotting models
----------------------------------------
- Works immediately with ANY phrase — "sterling", "hey sterling", etc.
- No training, no model download beyond what Whisper already needs.
- Accurate because it uses the same engine as the full STT.
- Easily updated: change the phrase list in config.yaml, restart.
"""

import re
import time

import numpy as np
import pyaudio
from faster_whisper import WhisperModel

from utils.logger import setup_logger

logger = setup_logger("sterling.wake_word")

# ── Audio constants ──────────────────────────────────────────────────────────
_SAMPLE_RATE  = 16000
_CHUNK        = 512          # 32 ms per read — fast VAD response
_INT16_MAX    = 32768.0

# Minimum speech chunks before a transcription is attempted.
# Filters out very brief clicks / transient noise (~96 ms minimum)
_MIN_SPEECH_CHUNKS = 3


class WakeWordDetector:
    """
    Listens for any of the configured wake phrases using Whisper transcription.

    Usage — same interface as the previous openWakeWord wrapper::

        detector = WakeWordDetector(phrases=["hey sterling", "sterling"])
        detector.start()
        while True:
            if detector.listen():
                break   # wake phrase detected
        detector.stop()
    """

    def __init__(
        self,
        phrases: list[str] = None,
        model_size: str = "tiny",
        energy_threshold: float = 500.0,
        silence_duration: float = 0.6,
        max_buffer_seconds: float = 2.5,
    ):
        """
        Args:
            phrases:            Wake phrases to listen for (case-insensitive, substring
                                match). e.g. ["hey sterling", "sterling", "ster", "ling"]
            model_size:         Faster-Whisper model size for wake detection.
                                "tiny" (recommended) is ~150 MB and transcribes in <200 ms.
                                "base" is more accurate but slower.
            energy_threshold:   RMS amplitude threshold to distinguish speech from silence.
                                Raise if background noise triggers false positives.
                                Lower if Sterling misses quiet speech.
            silence_duration:   Seconds of silence after speech before transcription runs.
            max_buffer_seconds: Maximum audio to accumulate before forcing transcription.
        """
        if not phrases:
            phrases = ["hey sterling", "sterling"]

        # Normalise phrases once at startup
        self._phrases = [self._normalise(p) for p in phrases]
        self._model_size = model_size
        self._threshold = energy_threshold

        chunks_per_sec = _SAMPLE_RATE / _CHUNK
        self._silence_chunks_needed = int(silence_duration * chunks_per_sec)
        self._max_buffer_chunks     = int(max_buffer_seconds * chunks_per_sec)

        # Whisper model — loaded in start()
        self._model: WhisperModel = None

        # PyAudio stream — opened in start()
        self._pa     = None
        self._stream = None

        # VAD state machine
        self._state         = "WAITING"   # WAITING | RECORDING
        self._buffer:  list = []
        self._speech_chunks  = 0
        self._silence_chunks = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        """Load the Whisper model and open the microphone stream."""
        logger.info(
            f"Loading wake word engine — "
            f"model='{self._model_size}', "
            f"phrases={[p for p in self._phrases]}"
        )
        t0 = time.time()
        self._model = WhisperModel(
            self._model_size,
            device="cpu",
            compute_type="int8",
        )
        logger.info(f"  Whisper '{self._model_size}' loaded in {time.time()-t0:.1f}s")

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            rate=_SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=_CHUNK,
        )

        self._reset_state()
        logger.info(
            f"Wake word detector ready — "
            f"threshold={self._threshold:.0f}, "
            f"silence={self._silence_chunks_needed} chunks"
        )

    def pause(self):
        """
        Close the audio stream while keeping the Whisper model loaded.
        Call before opening any other input stream (recorder, monitor).
        The main loop must not call listen() while paused.
        """
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        self._reset_state()
        logger.debug("Wake word detector paused.")

    def resume(self):
        """
        Reopen the audio stream after pause().
        Call once all other input streams have been closed.
        """
        if self._pa and self._stream is None:
            self._stream = self._pa.open(
                rate=_SAMPLE_RATE,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=_CHUNK,
            )
            self._reset_state()
            logger.debug("Wake word detector resumed.")

    def stop(self):
        """Close the audio stream and release all resources."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._pa:
            self._pa.terminate()
            self._pa = None

        self._model = None
        logger.debug("Wake word detector stopped.")

    def start_energy_monitor(
        self,
        stop_event: "threading.Event",
        threshold_multiplier: float = 2.5,
        sustain_seconds: float = 0.4,
    ) -> "threading.Thread":
        """
        Spawn a lightweight interruption monitor that reuses the EXISTING
        wake word PyAudio stream — no second PyAudio instance is opened,
        so there is no CoreAudio mutex deadlock.

        Safe because listen() is never called concurrently: the main loop
        is inside _handle_interaction() while this monitor runs.

        When sustained speech is detected (user interrupting), sets
        stop_event so TTS kills afplay within its 50 ms polling cycle.

        Args:
            stop_event:           Event shared with TTS.speak_streaming().
            threshold_multiplier: Energy bar relative to the wake-word threshold.
                                  Higher = less sensitive to background noise.
            sustain_seconds:      How long speech must persist before triggering.
        """
        import threading
        threshold = self._threshold * threshold_multiplier
        needed    = max(1, int(sustain_seconds * _SAMPLE_RATE / _CHUNK))

        def _monitor():
            count = 0
            while not stop_event.is_set():
                if self._stream is None:
                    break
                try:
                    raw = self._stream.read(_CHUNK, exception_on_overflow=False)
                except Exception:
                    break
                energy = self._rms(raw)
                if energy > threshold:
                    count += 1
                    if count >= needed:
                        logger.info("Interruption detected — stopping speech.")
                        stop_event.set()
                        break
                else:
                    count = 0

        t = threading.Thread(target=_monitor, daemon=True)
        t.start()
        return t

    # ─────────────────────────────────────────────────────────────────────────
    # Detection — called in a tight loop from main.py
    # ─────────────────────────────────────────────────────────────────────────

    def listen(self, stop_event=None) -> bool:
        """
        Read one 32 ms audio chunk and advance the VAD state machine.
        When enough speech has been accumulated and silence detected,
        transcribes with Whisper and checks for wake phrases.

        Args:
            stop_event: Optional threading.Event. When set, the method skips
                        the expensive Whisper transcription and returns False
                        immediately — lets the caller exit in ~32 ms instead
                        of waiting up to 200 ms for a transcription to finish.

        Returns:
            True  — wake phrase detected.
            False — still listening (call again immediately).

        Raises:
            RuntimeError — if start() has not been called.
        """
        if self._stream is None or self._model is None:
            raise RuntimeError("WakeWordDetector.start() must be called before listen().")

        raw = self._stream.read(_CHUNK, exception_on_overflow=False)
        energy = self._rms(raw)

        if self._state == "WAITING":
            if energy > self._threshold:
                # Speech onset — begin accumulating
                self._state = "RECORDING"
                self._buffer.append(raw)
                self._speech_chunks  = 1
                self._silence_chunks = 0

        elif self._state == "RECORDING":
            self._buffer.append(raw)

            if energy > self._threshold:
                self._speech_chunks  += 1
                self._silence_chunks  = 0
            else:
                self._silence_chunks += 1

            should_transcribe = (
                self._silence_chunks >= self._silence_chunks_needed
                or len(self._buffer) >= self._max_buffer_chunks
            )

            if should_transcribe:
                # If the caller has signalled stop, skip the ~200 ms Whisper
                # transcription — the result doesn't matter any more.
                if stop_event is not None and stop_event.is_set():
                    self._reset_state()
                    return False
                detected = self._transcribe_and_check()
                self._reset_state()
                return detected

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────────

    def _transcribe_and_check(self) -> bool:
        """Run Whisper on the accumulated buffer and check for wake phrases."""
        if self._speech_chunks < _MIN_SPEECH_CHUNKS:
            # Too short to be intentional speech — skip
            return False

        # Convert raw bytes → float32 [-1, 1]
        audio = np.frombuffer(b"".join(self._buffer), dtype=np.int16)
        audio = audio.astype(np.float32) / _INT16_MAX

        try:
            segments, _ = self._model.transcribe(
                audio,
                language="en",
                beam_size=3,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 200, "speech_pad_ms": 100},
                condition_on_previous_text=False,
            )
            text = " ".join(s.text for s in segments).strip()
        except Exception as e:
            logger.debug(f"Wake transcription error: {e}")
            return False

        if not text:
            return False

        text_norm = self._normalise(text)
        logger.debug(f"Wake heard: {text!r}  →  normalised: {text_norm!r}")

        for phrase in self._phrases:
            if phrase in text_norm:
                logger.info(f"Wake word detected!  heard={text!r}  matched={phrase!r}")
                return True

        return False

    def _reset_state(self):
        self._state         = "WAITING"
        self._buffer        = []
        self._speech_chunks  = 0
        self._silence_chunks = 0

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, strip punctuation/symbols. Used for phrase matching."""
        return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

    @staticmethod
    def _rms(raw: bytes) -> float:
        chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
