"""The capability ports, satisfied by delegating to one `LLMProvider`.

Three adapters that hold a provider and the name of a model to ask it for.
Nothing here names a backend or knows how one is reached — swapping LM Studio
for anything else changes `composition`, and these are untouched.

They are thin on purpose, and still worth having: the ports above them are
shaped the way the core wants to ask (`embed_documents(texts)`), while the
provider is shaped the way a backend is addressed (`embed_documents(model,
texts)`). Binding a model name to a capability is the whole job.
"""

from __future__ import annotations

from typing import Sequence

from domain.constants import OCR_INSTRUCTION, PROMPT_KEY_OCR
from ports.outbound import LLMProvider, PromptLibrary


class ProviderEmbeddings:
    """Implements `EmbeddingModel`."""

    def __init__(self, provider: LLMProvider, model: str):
        self._provider = provider
        self._model = model

    def embed_query(self, text: str) -> list[float]:
        return self._provider.embed_query(self._model, text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._provider.embed_documents(self._model, texts)

    def dimension(self) -> int:
        return self._provider.embedding_dimension(self._model)


class ProviderOcr:
    """Implements `OcrModel` using a vision-capable chat model."""

    def __init__(self, provider: LLMProvider, model: str, prompts: PromptLibrary):
        self._provider = provider
        self._model = model
        self._prompts = prompts

    @property
    def model_id(self) -> str:
        return self._model

    def read_page_image(self, jpeg: bytes) -> str:
        # Read per page rather than once per ingest, which costs nothing — the
        # library caches — and means the instruction cannot be stale for a run
        # that has been going since before it was edited.
        return self._provider.ocr(
            self._model,
            jpeg,
            self._prompts.text(PROMPT_KEY_OCR, OCR_INSTRUCTION),
        )


class ProviderReranker:
    """Implements `Reranker`."""

    def __init__(self, provider: LLMProvider, model: str):
        self._provider = provider
        self._model = model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return self._provider.rerank(self._model, query, documents)
