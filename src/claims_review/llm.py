"""A small LLM interface and the Groq implementation. Tests use a fake LLM and never reach the network."""
import os
from typing import Protocol

DEFAULT_MODEL = "openai/gpt-oss-120b"
KEY_VARIABLE = "GROQ_API_KEY"


class MissingAPIKeyError(RuntimeError):
    """Raised when the Groq API key is not set. The message names the variable, never a value."""


class LLM(Protocol):
    """Anything that turns chat messages into one text reply."""

    def complete(self, messages: list[dict]) -> str:
        ...


class GroqLLM:
    """Chat completions through the groq SDK: temperature 0, 60 second timeout.

    The key comes from the GROQ_API_KEY environment variable and the model from GROQ_MODEL
    (default openai/gpt-oss-120b). The key is never stored on this object or printed.
    """
    temperature = 0

    def __init__(self, model: str | None = None, timeout: float = 60.0, client=None):
        self.model = model or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        if client is None:
            key = (os.environ.get(KEY_VARIABLE) or "").strip()
            if not key:
                raise MissingAPIKeyError(
                    f"The environment variable {KEY_VARIABLE} is not set. Export it in your shell, "
                    "for example with `read -s` so the value is not echoed, then run again.")
            from groq import Groq
            client = Groq(api_key=key, timeout=timeout)
        self._client = client

    def __repr__(self) -> str:
        return f"GroqLLM(model={self.model!r}, timeout={self.timeout})"

    def complete(self, messages: list[dict]) -> str:
        """Send the messages and return the reply text ('' if the model returned no content)."""
        response = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, timeout=self.timeout)
        return response.choices[0].message.content or ""
