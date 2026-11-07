"""LLM provider: local llama.cpp or OpenAI-compatible API."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from config import Settings

logger = logging.getLogger(__name__)


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Create the chat model for the configured mode.

    Args:
        settings: Application settings.

    Returns:
        A LangChain chat model.

    Raises:
        ValueError: If required settings of the mode are missing.
    """
    if settings.llm_mode == "llamacpp":
        if not settings.llamacpp_model_path:
            raise ValueError("llamacpp 模式需要设置 NURSERY_LLAMACPP_MODEL_PATH")
        from langchain_community.chat_models import ChatLlamaCpp

        logger.info("Loading llama.cpp model: %s", settings.llamacpp_model_path)
        return ChatLlamaCpp(
            model_path=settings.llamacpp_model_path,
            n_ctx=settings.llamacpp_n_ctx,
            n_gpu_layers=settings.llamacpp_n_gpu_layers,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            verbose=False,
        )

    if not settings.api_model:
        raise ValueError("api 模式需要设置 NURSERY_API_MODEL（以及 NURSERY_API_BASE_URL / NURSERY_API_KEY）")
    from langchain_openai import ChatOpenAI

    logger.info("Using API model at %s", settings.api_base_url)
    return ChatOpenAI(
        model=settings.api_model,
        base_url=settings.api_base_url,
        api_key=settings.api_key or "EMPTY",
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


class LLMProvider:
    """Lazily creates and caches the chat model (thread-safe).

    Replacing the LLM only requires changing :func:`build_chat_model` or
    passing an object with ``get()`` to the planner.
    """

    def __init__(self, settings: Settings, model: Optional[BaseChatModel] = None) -> None:
        """Initialise the provider.

        Args:
            settings: Application settings.
            model: Optional pre-built model (used by tests).
        """
        self._settings = settings
        self._model = model
        self._lock = threading.Lock()

    def get(self) -> BaseChatModel:
        """Return the chat model, building it on first use."""
        with self._lock:
            if self._model is None:
                self._model = build_chat_model(self._settings)
            return self._model
