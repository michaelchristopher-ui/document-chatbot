from __future__ import annotations

from typing import Sequence

from pymilvus import DataType, MilvusClient

from adapters.outbound.vector_stores.constants import (
    BATCH_SIZE,
    COLLECTION_NAME,
    MATCH_ALL,
    METRIC_TYPE,
    OUTPUT_FIELDS,
    SCALAR_INDEX_FIELDS,
    SCALAR_INDEX_TYPE,
    SECTION_MAX,
    SOURCE_FILE_MAX,
    STRATEGY_MAX,
    TEXT_MAX,
    VECTOR_INDEX_TYPE,
)
from domain.filters import MetadataFilter
from domain.models import ChunkMetadata, EmbeddedChunk, Passage


def _fit(value: str, limit: int) -> str:
    """Trim to a field's byte budget. Lite ignores `max_length`; a Milvus server does not."""
    encoded = value.encode()
    return value if len(encoded) <= limit else encoded[:limit].decode(errors="ignore")


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _to_expression(where: MetadataFilter) -> str:
    """Compile a filter to a Milvus boolean expression; "" means unscoped."""
    clauses = []
    for field, values in (
        ("source_file", where.source_files),
        ("section", where.sections),
        ("strategy", where.strategies),
    ):
        if values:
            clauses.append(f"{field} in [{', '.join(_quote(v) for v in values)}]")
    if where.pages:
        first, last = where.pages
        clauses.append(f"page >= {int(first)} and page <= {int(last)}")
    return " and ".join(clauses)


def _to_passage(entity: dict, score: float | None = None) -> Passage:
    return Passage(
        text=entity["text"],
        page=entity["page"],
        source_file=entity["source_file"],
        strategy=entity.get("strategy", ""),
        metadata=ChunkMetadata(
            index=entity.get("chunk_index", -1),
            section=entity.get("section", ""),
            start=entity.get("start_char", -1),
            end=entity.get("end_char", -1),
        ),
        score=score,
    )


class MilvusStore:
    """Implements `VectorStore` on Milvus.

    `uri` is a local file path for milvus-lite, or a `http://host:port` server.
    """

    def __init__(self, uri: str, collection: str = COLLECTION_NAME):
        self._client = MilvusClient(uri)
        self._collection = collection

    def ensure_ready(self, dimension: int) -> None:
        if self._client.has_collection(self._collection):
            return

        # Declared field by field rather than via `dimension=`: scoping a search
        # needs real scalar fields, and a dynamic field cannot carry an index.
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        # Without auto_id the primary key must be supplied on every insert, and
        # inserts are rejected outright.
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field("text", DataType.VARCHAR, max_length=TEXT_MAX)
        schema.add_field("page", DataType.INT64)
        schema.add_field("source_file", DataType.VARCHAR, max_length=SOURCE_FILE_MAX)
        schema.add_field("strategy", DataType.VARCHAR, max_length=STRATEGY_MAX)
        schema.add_field("section", DataType.VARCHAR, max_length=SECTION_MAX)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("start_char", DataType.INT64)
        schema.add_field("end_char", DataType.INT64)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type=VECTOR_INDEX_TYPE, metric_type=METRIC_TYPE
        )
        for field in SCALAR_INDEX_FIELDS:
            index_params.add_index(field_name=field, index_type=SCALAR_INDEX_TYPE)

        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )

    def reset(self) -> None:
        if self._client.has_collection(self._collection):
            self._client.drop_collection(self._collection)

    def add(self, embedded: Sequence[EmbeddedChunk]) -> None:
        if not embedded:
            return
        self._client.insert(
            collection_name=self._collection,
            data=[
                {
                    "vector": item.vector,
                    "text": _fit(item.chunk.text, TEXT_MAX),
                    "page": item.chunk.page,
                    "source_file": _fit(item.chunk.source_file, SOURCE_FILE_MAX),
                    "strategy": _fit(item.chunk.strategy, STRATEGY_MAX),
                    "section": _fit(item.chunk.metadata.section, SECTION_MAX),
                    "chunk_index": item.chunk.metadata.index,
                    "start_char": item.chunk.metadata.start,
                    "end_char": item.chunk.metadata.end,
                }
                for item in embedded
            ],
        )

    def remove(self, source_file: str) -> None:
        if not self._client.has_collection(self._collection):
            return
        self._client.delete(
            collection_name=self._collection,
            filter=f"source_file == {_quote(source_file)}",
        )

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        where: MetadataFilter | None = None,
    ) -> list[Passage]:
        results = self._client.search(
            collection_name=self._collection,
            data=[list(vector)],
            limit=limit,
            filter=_to_expression(where) if where else "",
            output_fields=OUTPUT_FIELDS,
        )
        # The collection is indexed `METRIC_TYPE`, so `distance` is already the
        # similarity the port promises — larger is closer, 1.0 is identical.
        return [_to_passage(hit["entity"], hit["distance"]) for hit in results[0]]

    def max_similarity(self, vectors: Sequence[Sequence[float]]) -> list[float]:
        if not vectors:
            return []
        try:
            results = self._client.search(
                collection_name=self._collection,
                data=[list(v) for v in vectors],
                limit=1,
                output_fields=[],
            )
        except Exception:
            # Nothing indexed yet — nothing can be a duplicate.
            return [0.0] * len(vectors)
        return [hits[0]["distance"] if hits else 0.0 for hits in results]

    def all_passages(self) -> list[Passage]:
        """Every stored passage, paged out — a plain query is capped at 16384 rows."""
        if not self._client.has_collection(self._collection):
            return []

        iterator = self._client.query_iterator(
            collection_name=self._collection,
            batch_size=BATCH_SIZE,
            filter=MATCH_ALL,
            output_fields=OUTPUT_FIELDS,
        )
        passages: list[Passage] = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                passages.extend(_to_passage(row) for row in batch)
        finally:
            iterator.close()
        return passages
