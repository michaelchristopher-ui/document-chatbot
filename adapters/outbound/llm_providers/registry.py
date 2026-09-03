"""Backend selection for `ports.outbound.LLMProvider` and `ports.outbound.ModelRuntime`.

This is the only module that knows which inference servers exist. Swapping one
for another is a config change (`LLM_BACKEND`), and adding one is three steps:

1. write a provider satisfying `ports.outbound.LLMProvider` — and
   `adapters.outbound.agents.langgraph.ChatModelProvider` with it, since the
   agent drives a model object rather than a call through one;
2. write a runtime satisfying `ports.outbound.ModelRuntime`;
3. register both under one name in `_BACKENDS` below, and the same name in
   `constants.BACKENDS` with the address and models that go with it.

A backend need not be able to answer everything to be worth registering:
`RERANK_BACKEND` picks one for reranking alone, which is how `llamacpp` is
normally reached — see `config.Config.rerank_endpoint`.

Builders take a base URL because that is all every backend so far needs;
anything else (a credential, a timeout) belongs inside the adapter, read from
the environment, so this signature stays stable — `VMLXProvider` reads
`VMLX_API_KEY` that way.

Names are matched by value against `constants.BACKENDS`, which holds each one's
address, label and recommended models and imports nothing from here — the same
arrangement `VECTOR_BACKEND` and `vector_stores` are in.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from adapters.outbound.llm_providers.constants import LLAMA_CPP, LM_STUDIO, VMLX
from ports.outbound import LLMProvider, ModelRuntime


class _Builders(NamedTuple):
    provider: Callable[[str], LLMProvider]
    runtime: Callable[[str], ModelRuntime]


def _lm_studio_provider(base_url: str) -> LLMProvider:
    # Imported inside the builder so an unselected backend's dependencies never
    # have to be importable — this module must stay free of third-party imports.
    from adapters.outbound.llm_providers.lmstudio import LMStudioProvider

    return LMStudioProvider(base_url)


def _lm_studio_runtime(base_url: str) -> ModelRuntime:
    from adapters.outbound.model_runtimes.lmstudio import LmsRuntime

    return LmsRuntime(base_url)


def _vmlx_provider(base_url: str) -> LLMProvider:
    from adapters.outbound.llm_providers.vmlx import VMLXProvider

    return VMLXProvider(base_url)


def _vmlx_runtime(base_url: str) -> ModelRuntime:
    from adapters.outbound.model_runtimes.vmlx import VmlxRuntime

    return VmlxRuntime(base_url)


def _llama_cpp_provider(base_url: str) -> LLMProvider:
    from adapters.outbound.llm_providers.llama_cpp import LlamaCppProvider

    return LlamaCppProvider(base_url)


def _llama_cpp_runtime(base_url: str) -> ModelRuntime:
    from adapters.outbound.model_runtimes.llama_cpp import LlamaCppRuntime

    return LlamaCppRuntime(base_url)


_BACKENDS: dict[str, _Builders] = {
    LM_STUDIO: _Builders(_lm_studio_provider, _lm_studio_runtime),
    VMLX: _Builders(_vmlx_provider, _vmlx_runtime),
    # Registered like any other, and reached mostly as `RERANK_BACKEND`: one
    # llama-server holds one model, so it answers for a whole app only when
    # that model is the chat model. See `adapters.outbound.llm_providers.llama_cpp`.
    LLAMA_CPP: _Builders(_llama_cpp_provider, _llama_cpp_runtime),
}


def available_backends() -> tuple[str, ...]:
    return tuple(_BACKENDS)


def create_provider(backend: str, base_url: str) -> LLMProvider:
    """Build the configured provider, pointed at `base_url`.

    The returned object also satisfies `ChatModelProvider`, which is not on the
    port: it is declared beside `LangGraphAgent` to keep the LangChain type out
    of `ports.outbound`, and every provider registered here answers it.
    """
    return _builders(backend).provider(base_url)


def create_model_runtime(backend: str, base_url: str) -> ModelRuntime:
    """Build the configured runtime — how the setup screen sees installed models."""
    return _builders(backend).runtime(base_url)


def _builders(backend: str) -> _Builders:
    try:
        return _BACKENDS[backend]
    except KeyError:
        known = ", ".join(available_backends())
        raise ValueError(
            f"Unknown inference backend {backend!r}. Available: {known}."
        ) from None
