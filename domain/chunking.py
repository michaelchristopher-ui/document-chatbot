"""Chunking strategies.

Kept free of third-party imports by taking the two capabilities it cannot
implement itself — recursive text splitting and embedding — as plain callables,
supplied by the composition root.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable, Sequence

from domain.constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MIN_TEXT_LEN,
    SEMANTIC_SIMILARITY_THRESHOLD,
    SEMANTIC_WINDOW_SIZE,
    _ABBREV_MAP,
    _ABBREV_RESTORE,
    _HEADER_RE,
    _SENTENCE_RE,
)
from domain.models import Chunk, ChunkingStrategy, ChunkMetadata


SplitText = Callable[[str], list[str]]
EmbedTexts = Callable[[Sequence[str]], list[list[float]]]
Chunker = Callable[[str, int, str], list[Chunk]]


def _locate(text: str, chunk_text: str, cursor: int) -> tuple[int, int]:
    """Character span of `chunk_text` in `text` at or after `cursor`; (-1, -1) if absent."""
    start = text.find(chunk_text, cursor)
    if start >= 0:
        return start, start + len(chunk_text)

    # Strategies that rejoin sentences collapse the original whitespace, so the
    # chunk is no longer a verbatim substring — match on token boundaries instead.
    tokens = chunk_text.split()
    if not tokens:
        return -1, -1
    match = re.compile(r"\s+".join(re.escape(token) for token in tokens)).search(text, cursor)
    return (match.start(), match.end()) if match else (-1, -1)


def _finalize(chunks: list[Chunk], text: str) -> list[Chunk]:
    """Stamp reading-order index and page offsets onto otherwise finished chunks.

    Called last by every strategy, so a chunk that passed through a nested
    strategy still ends up numbered and located against the whole page.
    """
    located = []
    cursor = 0
    for index, chunk in enumerate(chunks):
        start, end = _locate(text, chunk.text, cursor)
        if start >= 0:
            # Advance by one rather than to `end`: fixed-size windows overlap.
            cursor = start + 1
        located.append(
            replace(chunk, metadata=replace(chunk.metadata, index=index, start=start, end=end))
        )
    return located


def chunk_fixed(text: str, page: int, source_file: str) -> list[Chunk]:
    chunks = []
    start = 0
    while start < len(text):
        chunk_text = text[start : start + CHUNK_SIZE].strip()
        if chunk_text:
            chunks.append(Chunk(chunk_text, page, source_file, "fixed"))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return _finalize(chunks, text)


def header_of(section: str) -> str:
    """The heading a section opens with, or "" when it does not open with one.

    Recognises exactly what `split_on_headers` splits on, so every section it
    produces except a leading preamble reports the header it was cut at.
    """
    first_line = section.lstrip().split("\n", 1)[0].strip()
    match = _HEADER_RE.match(first_line)
    return first_line[: match.end()].strip() if match else ""


def split_on_headers(text: str) -> list[str]:
    boundaries = [m.start() for m in _HEADER_RE.finditer(text)]
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + boundaries
    boundaries.append(len(text))
    return [
        text[boundaries[i]:boundaries[i + 1]].strip()
        for i in range(len(boundaries) - 1)
        if text[boundaries[i]:boundaries[i + 1]].strip()
    ]


def chunk_recursive(text: str, page: int, source_file: str, split: SplitText) -> list[Chunk]:
    sections = split_on_headers(text) or [text]
    chunks = []
    for section in sections:
        metadata = ChunkMetadata(section=header_of(section))
        for piece in split(section):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(piece, page, source_file, "recursive", metadata))
    return _finalize(chunks, text)


def split_sentences(text: str) -> list[str]:
    for abbrev, token in _ABBREV_MAP.items():
        text = text.replace(abbrev, token)
    sentences = _SENTENCE_RE.split(text)
    for restore_token, original in _ABBREV_RESTORE.items():
        sentences = [s.replace(restore_token, original) for s in sentences]
    return [s.strip() for s in sentences if s.strip()]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _compute_breakpoints(vecs: list[list[float]]) -> set[int]:
    n = len(vecs)
    w = SEMANTIC_WINDOW_SIZE
    breakpoints = set()
    dim = len(vecs[0])
    for i in range(w, n - w):
        before = vecs[i - w : i]
        after = vecs[i : i + w]
        before_mean = [sum(v[d] for v in before) / len(before) for d in range(dim)]
        after_mean = [sum(v[d] for v in after) / len(after) for d in range(dim)]
        if cosine_similarity(before_mean, after_mean) < SEMANTIC_SIMILARITY_THRESHOLD:
            breakpoints.add(i)
    return breakpoints


def chunk_semantic(text: str, page: int, source_file: str, embed: EmbedTexts) -> list[Chunk]:
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return chunk_fixed(text, page, source_file)

    breakpoints = _compute_breakpoints(embed(sentences))

    chunks: list[Chunk] = []
    current: list[str] = []
    for i, sentence in enumerate(sentences):
        if i in breakpoints and current:
            chunk_text = " ".join(current).strip()
            if len(chunk_text) >= MIN_TEXT_LEN:
                chunks.append(Chunk(chunk_text, page, source_file, "semantic"))
                current = [sentence]
            else:
                current.append(sentence)
        else:
            current.append(sentence)

    if current:
        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append(Chunk(chunk_text, page, source_file, "semantic"))

    final: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.text) > CHUNK_SIZE * 3:
            final.extend(chunk_fixed(chunk.text, page, source_file))
        else:
            final.append(chunk)

    # Nested `chunk_fixed` numbered its output against the oversized chunk it
    # split; finalizing here renumbers the page as a whole.
    return _finalize(final, text) if final else chunk_fixed(text, page, source_file)


def build_chunker(
    strategy: ChunkingStrategy,
    split: SplitText | None = None,
    embed: EmbedTexts | None = None,
) -> Chunker:
    """Bind a strategy to its dependencies, yielding a (text, page, file) callable."""
    if strategy == "fixed":
        return chunk_fixed
    if strategy == "recursive":
        if split is None:
            raise ValueError("a text splitter is required for recursive chunking")
        return lambda text, page, source: chunk_recursive(text, page, source, split)
    if strategy == "semantic":
        if embed is None:
            raise ValueError("an embedder is required for semantic chunking")
        return lambda text, page, source: chunk_semantic(text, page, source, embed)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
