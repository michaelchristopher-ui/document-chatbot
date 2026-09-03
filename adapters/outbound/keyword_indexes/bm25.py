from __future__ import annotations

from typing import Sequence

from rank_bm25 import BM25Okapi

from domain.filters import MetadataFilter
from domain.models import Passage


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class Bm25Index:
    """Implements `KeywordIndex` with an in-memory BM25Okapi index."""

    def __init__(self) -> None:
        self._passages: list[Passage] = []
        self._index: BM25Okapi | None = None

    def index(self, passages: Sequence[Passage]) -> None:
        self._passages = list(passages)
        self._index = BM25Okapi([_tokenize(p.text) for p in self._passages]) if self._passages else None

    def search(self, query: str, limit: int, where: MetadataFilter | None = None) -> list[Passage]:
        if self._index is None:
            return []
        scores = self._index.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Filtering while walking the ranked list, rather than after truncating
        # it, keeps a scoped search from returning fewer than `limit` matches.
        hits: list[Passage] = []
        for i in ranked:
            passage = self._passages[i]
            if where and not where.matches(passage):
                continue
            hits.append(passage)
            if len(hits) == limit:
                break
        return hits

    @property
    def is_empty(self) -> bool:
        return self._index is None
