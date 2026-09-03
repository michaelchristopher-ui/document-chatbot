"""Constants for `adapters.outbound.interaction_logs`."""

from __future__ import annotations

TURNS = "turns"
SEARCHES = "searches"
RETRIEVALS = "retrievals"
JUDGEMENTS = "judgements"

# `AUTOINCREMENT` because reads order by `id` as a stand-in for chronology, and a
# rowid SQLite reused would drop an old turn at the end of a time series. Nothing
# deletes rows today, but the ordering should not depend on that staying true.
#
# The nullable columns are nullable on purpose. A turn abandoned before its first
# token has no `first_token_ms`; a backend that reports no usage leaves both token
# counts empty; a passage that never reached the ledger has no `citation_index`.
# Storing zeros instead would let unmeasured turns average into the statistics.
#
# What is deliberately absent: passage text, which lives in the vector store and
# would make this a second copy of the corpus, and any per-turn count of searches
# or citations, which the child tables already answer and which would be a second
# truth free to drift from them.
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TURNS} (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id         TEXT    NOT NULL,
    created_at        TEXT    NOT NULL,
    question          TEXT    NOT NULL,
    answer            TEXT    NOT NULL,
    latency_ms        INTEGER NOT NULL,
    first_token_ms    INTEGER,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    chat_model        TEXT    NOT NULL,
    embed_model       TEXT    NOT NULL,
    reranker_model    TEXT    NOT NULL,
    chunking_strategy TEXT    NOT NULL,
    vector_backend    TEXT    NOT NULL,
    error             TEXT
);

CREATE TABLE IF NOT EXISTS {SEARCHES} (
    turn_id      INTEGER NOT NULL REFERENCES {TURNS}(id) ON DELETE CASCADE,
    search_index INTEGER NOT NULL,
    query        TEXT    NOT NULL,
    result_count INTEGER NOT NULL,
    PRIMARY KEY (turn_id, search_index)
);

CREATE TABLE IF NOT EXISTS {RETRIEVALS} (
    turn_id        INTEGER NOT NULL REFERENCES {TURNS}(id) ON DELETE CASCADE,
    search_index   INTEGER NOT NULL,
    rank           INTEGER NOT NULL,
    source_file    TEXT    NOT NULL,
    page           INTEGER NOT NULL,
    chunk_index    INTEGER NOT NULL,
    citation_index INTEGER,
    cited          INTEGER NOT NULL,
    score          REAL,
    PRIMARY KEY (turn_id, search_index, rank)
);

CREATE TABLE IF NOT EXISTS {JUDGEMENTS} (
    turn_id      INTEGER PRIMARY KEY REFERENCES {TURNS}(id) ON DELETE CASCADE,
    faithfulness REAL    NOT NULL,
    unsupported  TEXT    NOT NULL,
    model        TEXT    NOT NULL,
    judged_at    TEXT    NOT NULL
);
"""

# Columns added after the first version of this file shipped. SQLite cannot add
# them through `CREATE TABLE IF NOT EXISTS`, and a log that already holds answered
# turns must not be dropped to gain a column — so they are added in place, once.
_ADDED_COLUMNS = (
    (RETRIEVALS, "score", "REAL"),
    (RETRIEVALS, "text", "TEXT"),
    # Where each arm of the hybrid search had the passage before fusion, and where
    # fusion put it. NULL on every row written before this, which
    # `RetrievalRecord.arm` reads as unattributed rather than as an empty arm.
    (RETRIEVALS, "keyword_rank", "INTEGER"),
    (RETRIEVALS, "dense_rank", "INTEGER"),
    (RETRIEVALS, "fused_rank", "INTEGER"),
)

_INSERT_TURN = f"""
INSERT INTO {TURNS} (
    thread_id, created_at, question, answer, latency_ms, first_token_ms,
    prompt_tokens, completion_tokens, chat_model, embed_model, reranker_model,
    chunking_strategy, vector_backend, error
) VALUES (
    :thread_id, :created_at, :question, :answer, :latency_ms, :first_token_ms,
    :prompt_tokens, :completion_tokens, :chat_model, :embed_model, :reranker_model,
    :chunking_strategy, :vector_backend, :error
)
"""

_INSERT_SEARCH = f"""
INSERT INTO {SEARCHES} (turn_id, search_index, query, result_count)
VALUES (?, ?, ?, ?)
"""

# Named rather than positional: at thirteen columns a silent shift between the
# name list and the value list is a real hazard, and `_INSERT_TURN` already
# establishes the style.
_INSERT_RETRIEVAL = f"""
INSERT INTO {RETRIEVALS} (
    turn_id, search_index, rank, source_file, page, chunk_index, citation_index,
    cited, score, text, keyword_rank, dense_rank, fused_rank
) VALUES (
    :turn_id, :search_index, :rank, :source_file, :page, :chunk_index,
    :citation_index, :cited, :score, :text, :keyword_rank, :dense_rank, :fused_rank
)
"""

# Re-judging replaces: a judgement is one model's current reading of a turn, and
# two readings by the same judge are not two facts worth keeping.
_UPSERT_JUDGEMENT = f"""
INSERT INTO {JUDGEMENTS} (turn_id, faithfulness, unsupported, model, judged_at)
VALUES (:turn_id, :faithfulness, :unsupported, :model, :judged_at)
ON CONFLICT(turn_id) DO UPDATE SET
    faithfulness = excluded.faithfulness,
    unsupported  = excluded.unsupported,
    model        = excluded.model,
    judged_at    = excluded.judged_at
"""

# The window, shared by all three reads so they cannot disagree about which turns
# they are describing. `LIMIT` takes the newest rows; the outer `ORDER BY` puts
# them back in the order they happened, which is the order the charts plot.
_WINDOW = f"SELECT id FROM {TURNS} ORDER BY id DESC LIMIT ?"

_SELECT_TURNS = f"""
SELECT id, thread_id, created_at, question, answer, latency_ms, first_token_ms,
       prompt_tokens, completion_tokens, chat_model, embed_model, reranker_model,
       chunking_strategy, vector_backend, error
FROM {TURNS}
WHERE id IN ({_WINDOW})
ORDER BY id
"""

_SELECT_SEARCHES = f"""
SELECT turn_id, search_index, query, result_count
FROM {SEARCHES}
WHERE turn_id IN ({_WINDOW})
ORDER BY turn_id, search_index
"""

_SELECT_RETRIEVALS = f"""
SELECT turn_id, search_index, rank, source_file, page, chunk_index,
       citation_index, cited, score, text, keyword_rank, dense_rank, fused_rank
FROM {RETRIEVALS}
WHERE turn_id IN ({_WINDOW})
ORDER BY turn_id, search_index, rank
"""

_SELECT_JUDGEMENTS = f"""
SELECT turn_id, faithfulness, unsupported, model, judged_at
FROM {JUDGEMENTS}
WHERE turn_id IN ({_WINDOW})
"""

# Turns nothing has scored yet, newest first: the judge works backwards from the
# most recent, so a long backlog still says something about how the app runs now.
_SELECT_UNJUDGED = f"""
SELECT t.id
FROM {TURNS} t
LEFT JOIN {JUDGEMENTS} j ON j.turn_id = t.id
WHERE j.turn_id IS NULL AND t.answer <> ''
ORDER BY t.id DESC
LIMIT ?
"""
