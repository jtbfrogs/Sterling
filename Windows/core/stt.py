"""
Sterling Speech-to-Text
Uses Faster-Whisper (CTranslate2) for fast, accurate local transcription.
On M1 Mac: CPU device with int8 quantization is fastest and well-supported.

Model sizes (tradeoff speed vs accuracy):
  tiny   — 39M params,  ~390 MB,  very fast
  base   — 74M params,  ~580 MB,  fast  ← recommended for M1 8GB
  small  — 244M params, ~967 MB,  moderate
  medium — 769M params, ~3.1 GB,  slow on M1
"""

import numpy as np
from faster_whisper import WhisperModel
from utils.logger import setup_logger

logger = setup_logger("sterling.stt")


class STT:
    """
    Speech-to-Text wrapper around Faster-Whisper.
    Accepts float32 numpy audio arrays (normalized [-1, 1]).
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "en",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """
        Args:
            model_size:    Whisper model size: tiny | base | small | medium
            language:      ISO-639 language code. "en" for English.
                           Set to None for auto-detect (slightly slower).
            device:        "cpu" for M1 Mac. "cuda" for NVIDIA GPU builds.
            compute_type:  "int8" (fast, M1 optimized) | "float16" (GPU) | "float32"
        """
        self._language = language
        logger.info(f"Loading Faster-Whisper ({model_size}) on {device} with {compute_type}...")

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        logger.info("STT model loaded and ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: float32 numpy array, values in [-1.0, 1.0], sample rate 16000 Hz.

        Returns:
            Transcribed text string. Returns "" if audio is empty or no speech detected.
        """
        if audio is None or len(audio) == 0:
            logger.debug("STT received empty audio — skipping.")
            return ""

        logger.debug(f"Transcribing {len(audio) / 16000:.1f}s of audio...")

        try:
            segments, info = self._model.transcribe(
                audio,
                beam_size=5,
                language=self._language,
                vad_filter=True,           # Built-in VAD to filter out non-speech
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                },
                condition_on_previous_text=False,
            )

            text = " ".join(segment.text for segment in segments).strip()

            if text:
                logger.info(f"Transcribed: {text!r}")
            else:
                logger.debug("No speech content detected in audio.")

            return text

        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return ""
