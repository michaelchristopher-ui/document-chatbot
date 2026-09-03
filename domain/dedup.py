"""Three layers of duplicate detection, and the digest that precedes them.

`datasketch` is a pure computation library — no I/O — so it stays in the core
rather than hiding behind a port.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np
from datasketch import MinHash

from domain.constants import (
    DEDUP_COSINE_THRESHOLD,
    DEDUP_MINHASH_NUM_PERM,
    DEDUP_MINHASH_THRESHOLD,
)
from domain.models import Chunk


def content_digest(data: bytes) -> str:
    """A document's bytes as one value — the question the three layers precede.

    Layer 0, in effect, and the cheapest of the four: the others ask whether two
    documents say the same thing, this asks whether this is the same file. It is
    what an ingest checks before opening anything, so an unchanged document
    costs a read and a hash rather than a parse, an OCR pass and an embedding
    call.
    """
    return hashlib.sha256(data).hexdigest()


def document_signature(text: str) -> MinHash:
    m = MinHash(num_perm=DEDUP_MINHASH_NUM_PERM)
    for word in text.lower().split():
        m.update(word.encode())
    return m


def serialize_signature(signature: MinHash) -> bytes:
    """A signature as bytes, for a store that has to hand it back to a later run.

    The hash values alone. Everything else about a MinHash — the permutations,
    the seed — follows from `DEDUP_MINHASH_NUM_PERM`, which is a constant here
    rather than a per-document choice, so storing it would be storing the same
    thing once per document.
    """
    return signature.hashvalues.astype("<u8").tobytes()


def deserialize_signature(data: bytes) -> MinHash:
    """The inverse of `serialize_signature`, for comparing against a stored run."""
    return MinHash(
        num_perm=DEDUP_MINHASH_NUM_PERM, hashvalues=np.frombuffer(data, dtype="<u8")
    )


def find_near_duplicate(signature: MinHash, known: dict[str, MinHash]) -> str | None:
    """Return the name of a known document this one duplicates, if any."""
    for filename, stored in known.items():
        if signature.jaccard(stored) >= DEDUP_MINHASH_THRESHOLD:
            return filename
    return None


def dedup_exact(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Drop byte-identical chunks — catches overlapping windows within a document."""
    seen: set[str] = set()
    result: list[Chunk] = []
    for chunk in chunks:
        digest = hashlib.sha256(chunk.text.encode()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            result.append(chunk)
    return result


def drop_near_duplicates(
    chunks: Sequence[Chunk],
    vectors: Sequence[list[float]],
    similarities: Sequence[float],
) -> tuple[list[Chunk], list[list[float]]]:
    """Drop chunks too close to something already stored, given their best match."""
    keep_chunks: list[Chunk] = []
    keep_vectors: list[list[float]] = []
    for chunk, vector, similarity in zip(chunks, vectors, similarities):
        if similarity >= DEDUP_COSINE_THRESHOLD:
            continue
        keep_chunks.append(chunk)
        keep_vectors.append(vector)
    return keep_chunks, keep_vectors
