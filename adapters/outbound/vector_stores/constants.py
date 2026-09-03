"""Constants for `adapters.outbound.vector_stores`."""

from __future__ import annotations

# ── Backend names, as `VECTOR_BACKEND` spells them ────────────────────────────

MILVUS = "milvus"
MEMORY = "memory"


# ── Milvus ────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "documents"
BATCH_SIZE = 1000
MATCH_ALL = "source_file != ''"

# Every stored field is read back: scoped search filters in the engine, but the
# keyword index this also feeds filters in Python and needs the same metadata.
OUTPUT_FIELDS = [
    "text", "page", "source_file", "strategy", "section",
    "chunk_index", "start_char", "end_char",
]

TEXT_MAX = 65535
SOURCE_FILE_MAX = 512
SECTION_MAX = 512
STRATEGY_MAX = 32

# Index and metric names are the server's vocabulary and have to be spelled out,
# because pymilvus offers no constant to bind them to: `IndexType` and
# `MetricType` are v1 leftovers that predate INVERTED and COSINE alike, and
# `add_index` forwards whatever string it is handed. So nothing validates these
# until the create_index RPC, and a rename upstream would surface as a runtime
# error on a fresh collection rather than as an import that stops resolving.
# Naming them here is the most a client can do: one line to correct, not four.
VECTOR_INDEX_TYPE = "FLAT"
SCALAR_INDEX_TYPE = "INVERTED"
METRIC_TYPE = "COSINE"

# Milvus Lite rejects STL_SORT, BITMAP and AUTOINDEX. INVERTED is accepted on
# both VARCHAR and INT64, which covers every field a search can be scoped by.
SCALAR_INDEX_FIELDS = ("source_file", "page", "section", "strategy")
