"""A `VectorStore` with no database behind it at all.

It earns its place twice over: it is the second implementation that keeps the
port honest — nothing above `ports.outbound.VectorStore` can assume Milvus — and
it lets ingestion and retrieval run with no service to install. Everything lives
in the process, so it starts empty on every run.

Search is brute force, which is fine at the scale a local document set reaches
and hopeless beyond it. Use it for tests and experiments, not for a corpus —
and note that it alone re-ingests from scratch every run, since an index this
never persists is one no ledger entry can be reused against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from domain.filters import MetadataFilter
from domain.models import EmbeddedChunk, Passage


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


@dataclass(frozen=True)
class _Entry:
    """A stored chunk, with its norm cached — every search divides by it."""

    passage: Passage
    vector: tuple[float, ...]
    norm: float


class InMemoryVectorStore:
    """Implements `VectorStore` over a plain list."""

    def __init__(self) -> None:
        self._entries: list[_Entry] = []

    def ensure_ready(self, dimension: int) -> None:
        """Nothing to create — a list needs no schema."""

    def reset(self) -> None:
        self._entries.clear()

    def add(self, embedded: Sequence[EmbeddedChunk]) -> None:
        self._entries.extend(
            _Entry(item.chunk.as_passage(), tuple(item.vector), _norm(item.vector))
            for item in embedded
        )

    def remove(self, source_file: str) -> None:
        self._entries = [
            entry for entry in self._entries
            if entry.passage.source_file != source_file
        ]

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        where: MetadataFilter | None = None,
    ) -> list[Passage]:
        candidates = [
            entry for entry in self._entries
            if where is None or where.matches(entry.passage)
        ]
        scored = self._similarities(vector, candidates)
        ranked = sorted(zip(candidates, scored), key=lambda pair: pair[1], reverse=True)
        return [
            replace(entry.passage, score=score) for entry, score in ranked[:limit]
        ]

    def max_similarity(self, vectors: Sequence[Sequence[float]]) -> list[float]:
        if not self._entries:
            return [0.0] * len(vectors)
        return [
            max(self._similarities(vector, self._entries), default=0.0)
            for vector in vectors
        ]

    def all_passages(self) -> list[Passage]:
        return [entry.passage for entry in self._entries]

    def _similarities(
        self, vector: Sequence[float], entries: Sequence[_Entry]
    ) -> list[float]:
        """Cosine similarity against each entry, in input order."""
        query_norm = _norm(vector)
        if not query_norm:
            return [0.0] * len(entries)
        return [
            sum(a * b for a, b in zip(vector, entry.vector)) / (query_norm * entry.norm)
            if entry.norm
            else 0.0
            for entry in entries
        ]
