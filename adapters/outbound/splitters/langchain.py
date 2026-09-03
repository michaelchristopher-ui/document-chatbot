from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from adapters.outbound.splitters.constants import SEPARATORS
from domain.constants import CHUNK_OVERLAP, CHUNK_SIZE


class RecursiveSplitter:
    """Supplies `domain.chunking.SplitText`, backed by LangChain's splitter."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self._splitter = RecursiveCharacterTextSplitter(
            separators=SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def __call__(self, text: str) -> list[str]:
        return self._splitter.split_text(text)
