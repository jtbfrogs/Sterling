"""
Sterling Audio Recorder
Captures microphone input using energy-based Voice Activity Detection (VAD).
Recording stops automatically after a configurable silence duration.
"""

import time
from typing import Optional

import pyaudio
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("sterling.audio")


class AudioRecorder:
    """
    Records audio from the default microphone and stops when sustained
    silence is detected. Returns a normalized float32 numpy array suitable
    for Faster-Whisper.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        silence_threshold: float = 500.0,
        silence_duration: float = 1.5,
        max_recording_seconds: float = 30.0,
    ):
        """
        Args:
            sample_rate:            Audio sample rate in Hz (16000 for Whisper).
            channels:               Number of audio channels (1 = mono).
            chunk_size:             Frames per PyAudio buffer read.
            silence_threshold:      RMS energy below which audio is considered silence.
                                    Tune for your microphone — raise if noisy, lower if sensitive.
            silence_duration:       Seconds of continuous silence before recording stops.
            max_recording_seconds:  Hard cap on recording length to prevent runaway recordings.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_frames_needed = int(silence_duration * sample_rate / chunk_size)
        self.max_frames = int(max_recording_seconds * sample_rate / chunk_size)

        self._pa     = pyaudio.PyAudio()
        self._stream = None   # persistent stream; opened by open_stream()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def record_until_silence(self) -> np.ndarray:
        """
        Open the microphone, record until silence is detected, then close.

        Returns:
            numpy.ndarray of float32 in range [-1.0, 1.0], suitable for Whisper.
            Returns an empty array if nothing was captured.
        """
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        logger.debug("Recording started — waiting for speech...")

        frames: list[bytes] = []
        silent_chunk_count = 0
        speech_detected = False
        total_chunks = 0

        try:
            while total_chunks < self.max_frames:
                raw = stream.read(self.chunk_size, exception_on_overflow=False)
                total_chunks += 1

                energy = self._rms_energy(raw)

                if energy > self.silence_threshold:
                    # Active speech
                    speech_detected = True
                    silent_chunk_count = 0
                    frames.append(raw)
                elif speech_detected:
                    # Silence after speech has started — count down
                    frames.append(raw)
                    silent_chunk_count += 1
                    if silent_chunk_count >= self.silence_frames_needed:
                        logger.debug("Silence detected — stopping recording.")
                        break
                # If no speech detected yet, keep waiting (don't accumulate silence)

        finally:
            stream.stop_stream()
            stream.close()

        if not frames or not speech_detected:
            logger.debug("No speech detected.")
            return np.array([], dtype=np.float32)

        # Concatenate and normalize to float32 [-1, 1]
        raw_audio = np.frombuffer(b"".join(frames), dtype=np.int16)
        return raw_audio.astype(np.float32) / 32768.0

    def open_stream(self):
        """
        Start the persistent input stream.
        First call allocates it; subsequent calls just restart it (~10 ms)
        instead of destroying and recreating (~200 ms).
        """
        if self._stream is None:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            logger.debug("Persistent input stream opened.")
        else:
            try:
                if not self._stream.is_active():
                    self._stream.start_stream()
                    logger.debug("Persistent input stream restarted.")
            except Exception:
                # Stream in bad state — recreate it
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                )
                logger.debug("Persistent input stream recreated.")

    def close_stream(self):
        """
        Suspend the persistent stream without destroying it.
        Uses stop_stream() so open_stream() can restart in ~10 ms.
        """
        if self._stream:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
            except Exception:
                pass
            logger.debug("Persistent input stream stopped.")

    def listen_for_speech(self, timeout_seconds: float) -> Optional[np.ndarray]:
        """
        Wait up to `timeout_seconds` for the user to start speaking, then
        record until silence. Everything happens on a single stream that opens
        and closes within this call — no stream is left dangling.

        Used by the conversation loop so Sterling keeps listening between
        turns without needing the wake word again.

        Returns:
            float32 numpy array [-1, 1] if speech captured, None if timed out.
        """
        # Reuse the persistent stream if open; otherwise open a temporary one.
        own_stream = self._stream is None
        stream     = self._stream if self._stream else self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        frames: list[bytes] = []
        speech_detected = False
        silent_chunks   = 0
        total_chunks    = 0
        deadline        = time.time() + timeout_seconds

        try:
            while True:
                try:
                    raw = stream.read(self.chunk_size, exception_on_overflow=False)
                except OSError:
                    # Stream closed externally (e.g. shutdown) — bail out cleanly
                    return None

                energy = self._rms_energy(raw)
                total_chunks += 1

                if not speech_detected:
                    if time.time() >= deadline:
                        return None
                    if energy > self.silence_threshold:
                        speech_detected = True
                        frames.append(raw)
                else:
                    frames.append(raw)
                    if energy <= self.silence_threshold:
                        silent_chunks += 1
                        if silent_chunks >= self.silence_frames_needed:
                            break
                    else:
                        silent_chunks = 0
                    if total_chunks >= self.max_frames:
                        break
        finally:
            # Only close if we opened a temporary stream
            if own_stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

        if not frames:
            return None

        raw_audio = np.frombuffer(b"".join(frames), dtype=np.int16)
        return raw_audio.astype(np.float32) / 32768.0

    def terminate(self):
        """Release all PyAudio resources. Call on shutdown."""
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()   # full close on final shutdown
            except Exception:
                pass
            self._stream = None
        self._pa.terminate()
        logger.debug("AudioRecorder terminated.")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rms_energy(raw_bytes: bytes) -> float:
        """Compute Root Mean Square energy of a raw int16 audio chunk."""
        chunk = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(chunk ** 2)))
