"""
Sterling LLM Interface
Wraps the Ollama Python client for local LLM inference.
Supports both standard (blocking) and streaming response modes.

Ollama must be running: `ollama serve`
Model must be pulled:   `ollama pull llama3.2`
"""

from typing import Generator
import ollama
from utils.logger import setup_logger

logger = setup_logger("sterling.llm")


class LLM:
    """
    Ollama-backed language model client.

    Supports:
        - chat(messages) → str                  (full response, blocking)
        - stream(messages) → Generator[str]     (token-by-token streaming)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 512,
        context_window: int = 4096,
    ):
        """
        Args:
            model:       Ollama model name. Must be pulled: `ollama pull <model>`
            base_url:    Ollama server URL. Default is local.
            temperature: Sampling temperature. 0.0 = deterministic, 1.0 = creative.
            max_tokens:  Maximum number of tokens in the response. Lower = shorter/faster.
        """
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._context_window = context_window
        self._client = ollama.Client(host=base_url)

        logger.info(f"LLM initialized — model: {model}, temperature: {temperature}, max_tokens: {max_tokens}, context_window: {context_window}")
        self._verify_model()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def chat(self, messages: list[dict]) -> str:
        """
        Send a full conversation and return the complete response text.
        Blocks until the LLM finishes generating.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": str}

        Returns:
            Response text string. Returns "" on failure.
        """
        try:
            logger.debug(f"Sending {len(messages)} messages to LLM...")
            response = self._client.chat(
                model=self._model,
                messages=messages,
                options={
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                    "num_ctx": self._context_window,
                },
            )
            text = response["message"]["content"]
            logger.debug(f"LLM response: {len(text)} chars")
            return text
        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            return ""

    def stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """
        Send a full conversation and yield response tokens as they arrive.
        Use this for sentence-streaming TTS to reduce perceived latency.

        Args:
            messages: List of {"role": ..., "content": ...}

        Yields:
            String chunks (partial tokens or words) as they arrive from the model.
        """
        try:
            logger.debug(f"Streaming {len(messages)} messages to LLM...")
            response_stream = self._client.chat(
                model=self._model,
                messages=messages,
                stream=True,
                options={
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                    "num_ctx": self._context_window,
                },
            )
            for chunk in response_stream:
                delta = chunk["message"]["content"]
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"LLM stream failed: {e}")
            return

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _verify_model(self):
        """Check that the requested model is available in Ollama."""
        try:
            models = self._client.list()
            available = [m["model"] for m in models.get("models", [])]
            # Normalize: ollama may store as "llama3.2:latest"
            base_names = [m.split(":")[0] for m in available]
            model_base = self._model.split(":")[0]

            if model_base not in base_names:
                logger.warning(
                    f"Model '{self._model}' not found in Ollama. "
                    f"Available: {available}. "
                    f"Run: ollama pull {self._model}"
                )
            else:
                logger.info(f"Model '{self._model}' confirmed available.")
        except Exception as e:
            logger.warning(f"Could not verify Ollama model availability: {e}")
            logger.warning("Make sure Ollama is running: ollama serve")
