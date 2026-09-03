"""Backend selection for `ports.outbound.VectorStore`.

This is the only module that knows which vector databases exist. Swapping one
for another is a config change (`VECTOR_BACKEND`), and adding one is two steps:

1. write an adapter satisfying `ports.outbound.VectorStore`;
2. register a builder for it in `_BUILDERS` below.

Builders take `(uri, collection)` because that is all every backend so far
needs; anything further (credentials, region, pool size) belongs inside the
adapter, read from the environment, so this signature stays stable.
"""

from __future__ import annotations

from typing import Callable

from adapters.outbound.vector_stores.constants import MEMORY, MILVUS
from ports.outbound import VectorStore


def _milvus(uri: str, collection: str) -> VectorStore:
    # Imported inside the builder so an unselected backend's SDK never has to be
    # installed — importing this module must stay free of third-party imports.
    from adapters.outbound.vector_stores.milvus import MilvusStore

    return MilvusStore(uri, collection)


def _memory(uri: str, collection: str) -> VectorStore:
    from adapters.outbound.vector_stores.memory import InMemoryVectorStore

    return InMemoryVectorStore()


_BUILDERS: dict[str, Callable[[str, str], VectorStore]] = {
    MILVUS: _milvus,
    MEMORY: _memory,
}


def available_backends() -> tuple[str, ...]:
    return tuple(_BUILDERS)


def create_vector_store(backend: str, uri: str, collection: str) -> VectorStore:
    """Build the configured store.

    `uri` is backend-specific: a local file path for milvus-lite, a server URL
    for a remote Milvus, ignored entirely by the in-memory store.
    """
    try:
        build = _BUILDERS[backend]
    except KeyError:
        known = ", ".join(available_backends())
        raise ValueError(
            f"Unknown vector backend {backend!r}. Available: {known}."
        ) from None
    return build(uri, collection)
