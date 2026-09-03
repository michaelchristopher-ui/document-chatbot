"""Answer cache page — a third driving adapter, over `ViewCacheStatistics` alone.

The whole page is about one distinction, so it is drawn as two sections rather
than one grid of numbers: what this *process* has counted since it started, and
what the *store* is holding right now. Mixing them would be the misleading
version — a hit rate that resets on deploy sitting beside an entry count that
does not, with nothing saying which is which.

Numbers this page deliberately does not show:

- Money or tokens saved. A hit is a model call not made, and what that call
  would have cost is unknowable after the fact: the answer it would have written
  might have been longer or shorter than the one being replayed. The Statistics
  page reports tokens actually spent, which is a fact.
- Why misses missed. `MissReason` and the near-miss similarity are per-lookup
  readings the library reports and these counters do not aggregate, so there is
  nothing honest to draw here yet. Recording them per turn in the interaction
  log is what would make the threshold tunable from evidence rather than from a
  hunch — see the note at the foot of this page.
"""

from __future__ import annotations

import streamlit as st

from domain.answers import CacheStatistics
from ports.inbound import ViewCacheStatistics


def _percent(share: float) -> str:
    return f"{share * 100:.1f}%"


def _duration(seconds: float) -> str:
    """The largest unit that keeps the number short, for a TTL rather than a wait."""
    if seconds >= 86_400:
        return f"{seconds / 86_400:.0f} d"
    if seconds >= 3_600:
        return f"{seconds / 3_600:.0f} h"
    if seconds >= 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


def _render_this_process(stats: CacheStatistics) -> None:
    """The counters, and the one that says whether the cache is earning its keep."""
    st.subheader("Since this process started")
    st.caption(
        "Counted in memory by the cache itself, so these reset on restart, are "
        "not shared between replicas, and can lose an increment when two "
        "questions are answered at the same moment. Read them, do not bill "
        "from them."
    )

    top = st.columns(4)
    top[0].metric("Questions looked up", f"{stats.lookups:,}")
    top[1].metric(
        "Served from cache",
        _percent(stats.hit_rate),
        help=(
            f"{stats.hits:,} of {stats.lookups:,} lookups. Each one is a model "
            "call, a retrieval and a rerank that did not happen."
        ),
    )
    top[2].metric(
        "Found by meaning",
        _percent(stats.semantic_hit_rate),
        help=(
            f"{stats.semantic_hits:,} hits that needed the embedding model to "
            "find them — the question had been asked before in different words. "
            "This is what the similarity search earns over a plain string "
            "comparison, and the number to watch when moving the threshold."
        ),
    )
    top[3].metric(
        "Found word for word",
        _percent(stats.exact_hit_rate),
        help=(
            f"{stats.exact_hits:,} hits on a question identical after "
            "normalisation. Free — no embedding call was made to find them."
        ),
    )

    bottom = st.columns(4)
    bottom[0].metric(
        "Answers written",
        f"{stats.stores:,}",
        help=(
            "Answers streamed in full and stored. Lower than the miss count "
            "whenever a reader navigated away mid-answer: a half-written answer "
            "is deliberately not cached."
        ),
    )
    bottom[1].metric("Misses", f"{stats.misses:,}")
    bottom[2].metric(
        "Cache errors",
        f"{stats.errors:,}",
        help=(
            "Lookups where the store or the embedder could not be reached. Each "
            "was answered the long way round instead — a cache that cannot be "
            "reached costs time, not correctness."
        ),
    )
    bottom[3].metric(
        "Error rate",
        _percent(stats.error_rate),
        help="Errors as a share of lookups. Anything but ~0% means Redis is unwell.",
    )

    if stats.errors and stats.error_rate > 0.1:
        st.warning(
            f"{_percent(stats.error_rate)} of lookups could not reach the cache. "
            "Answers are unaffected — every one of those was answered from the "
            "documents — but nothing is being cached while this lasts. Check that "
            "the Redis at `REDIS_URL` is up."
        )


def _render_store(stats: CacheStatistics) -> None:
    """What is actually in Redis, and the settings that decide what may be served."""
    st.subheader("In the store now")
    st.caption(
        "Read from Redis, so this includes answers written by earlier runs and "
        "by other replicas, and excludes any the store has already expired."
    )

    columns = st.columns(3)
    if stats.entries_known:
        columns[0].metric(
            "Answers held",
            f"{stats.entries:,}",
            help=(
                "For this index and chat model only. Changing either points the "
                "app at a different keyspace rather than at answers the new "
                "combination did not write."
            ),
        )
    else:
        columns[0].metric("Answers held", "—")
        columns[0].caption("Store unreachable — not the same as empty.")

    columns[1].metric(
        "Similarity threshold",
        f"{stats.threshold:.2f}",
        help=(
            "How close a question has to be to serve a stored answer, as a "
            "cosine. Strict on purpose: too loose and the cache answers a "
            "question nobody asked, which costs a wrong answer, while too tight "
            "only costs a model call."
        ),
    )
    columns[2].metric(
        "Answer lifetime",
        _duration(stats.ttl_seconds),
        help=(
            "How long an answer may be served for. A backstop rather than the "
            "mechanism: what actually invalidates an answer is the corpus behind "
            "it changing, and an ingest that indexes anything empties the cache."
        ),
    )


def render(cache: ViewCacheStatistics) -> None:
    st.title("Answer cache")
    st.caption(
        "A question close enough to one already answered is served from Redis "
        "without a model call. Only the first question in a conversation is "
        "cached — a follow-up means something only in the context of the thread "
        "it was asked in."
    )

    stats = cache.statistics()

    if not stats.lookups:
        st.info(
            "No questions looked up yet in this process. Ask something on the "
            "Chat page — the first of each kind pays for itself, and the second "
            "arrives instantly."
        )
        # The store still has something to say: entries written before this
        # process started are the reason a first lookup can hit at all.
        st.divider()
        _render_store(stats)
        return

    _render_this_process(stats)
    st.divider()
    _render_store(stats)
    st.divider()

    st.caption(
        "**What is not here yet.** Why a miss missed — how close the nearest "
        "stored answer came — is reported per lookup and not aggregated by these "
        "counters, so the threshold above cannot yet be tuned from evidence. "
        "Recording it against each turn in the interaction log is what would fix "
        "that, and would survive a restart as these numbers do not."
    )
