"""Statistics page — a second driving adapter, over `ViewStatistics` alone.

It reads turns and hands them to `domain.statistics`; the arithmetic is not this
file's business, and neither is the log's storage. What it does own is which
numbers are worth a reader's attention, and in what order.

Charts are Streamlit's own. Their colours come from the active theme, so light and
dark both work without this file naming a single hex value — one it named would be
right in one theme and wrong in the other.
"""

from __future__ import annotations

from typing import Optional, Sequence

import streamlit as st

from adapters.inbound.constants import (
    DEFAULT_BATCH,
    DEFAULT_WINDOW,
    MAX_BATCH,
    QUESTION_LABEL_CHARS,
    RECENT_SEARCHES,
    RECENT_TURNS,
    WINDOW_OPTIONS,
)
from domain.constants import (
    FAITHFUL_THRESHOLD,
    FIRST_TOKEN_SERIES,
    FLAT_RETRIEVAL,
    TOTAL_SERIES,
    TURN_AXIS,
    UNATTRIBUTED,
    WEAK_RETRIEVAL,
)
from domain.interactions import TurnRecord
from domain.statistics import (
    ArmTotals,
    Summary,
    arm_usage,
    outcome,
    by_configuration,
    latency_series,
    per_day,
    recent_searches,
    source_usage,
    summarise,
)
from ports.inbound import ScoreAnswers, ViewStatistics


# ── Formatting ────────────────────────────────────────────────────────────────

def _duration(milliseconds: int) -> str:
    """The largest unit that keeps the number short.

    A big local model answering from a cold cache takes minutes, and "618.1 s" is
    a number a reader has to divide before it means anything.
    """
    if not milliseconds:
        return "—"
    if milliseconds < 1000:
        return f"{milliseconds} ms"
    seconds = milliseconds / 1000
    if seconds < 90:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}m {remainder:02d}s"


def _percent_text(share: float) -> str:
    """A 0–1 share as a label, for anything that shows the number itself."""
    return f"{share * 100:.0f}%"


def _percent_number(share: float) -> float:
    """A 0–1 share on the 0–100 scale `_share_column` draws against."""
    return 100 * share


def _count(value: Optional[float], suffix: str = "") -> str:
    return "—" if value is None else f"{value:,.0f}{suffix}"


def _truncate(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit].rstrip() + "…"


def _share_column(label: str):
    """A share drawn as a bar, on the scale `_percent_number` puts it on.

    Defined once because the scale and the format have to agree: a column left on
    Streamlit's own `percent` format multiplies by a hundred again, and a share
    already scaled reads as 9,000%.
    """
    return st.column_config.ProgressColumn(
        # Whole percents — a share this noisy does not have two decimals of
        # precision to report.
        label, format="%.0f%%", min_value=0, max_value=100
    )


# ── Sections ──────────────────────────────────────────────────────────────────

def _render_headline(summary: Summary) -> None:
    """The four numbers worth reading first, then the four behind them."""
    top = st.columns(4)
    top[0].metric("Turns", f"{summary.turns:,}")
    top[1].metric(
        "Grounded answers",
        _percent_text(summary.grounded_share),
        help=(
            f"{summary.grounded_turns:,} of {summary.turns:,} answers cited at least "
            "one retrieved passage. The rest were refusals, or were written without "
            "the documents."
        ),
    )
    top[2].metric(
        "Median answer time",
        _duration(summary.median_latency_ms),
        help=f"95th percentile: {_duration(summary.p95_latency_ms)}",
    )
    top[3].metric(
        "Median time to first token",
        _duration(summary.median_first_token_ms),
        help=(
            "Counts the searches the model runs before it writes anything — the wait "
            "a reader actually sits through."
        ),
    )

    bottom = st.columns(4)
    bottom[0].metric(
        "Searches per turn",
        f"{summary.searches_per_turn:.1f}",
        help=f"{summary.searches:,} searches in total.",
    )
    bottom[1].metric(
        "Passages per turn",
        f"{summary.passages_per_turn:.1f}",
        help=f"{summary.retrieved:,} passages retrieved in total.",
    )
    bottom[2].metric(
        "Passages cited",
        _percent_text(summary.cited_share),
        help=(
            f"{summary.cited:,} of {summary.retrieved:,} retrieved passages were cited. "
            "Retrieval casts wide on purpose, so this is never near 100%."
        ),
    )
    bottom[3].metric(
        "Tokens",
        _count(summary.usage.total or None),
        help=(
            f"{summary.usage.prompt:,} prompt + {summary.usage.completion:,} completion, "
            f"over the {summary.turns_with_usage:,} turns whose backend reported usage."
            if summary.turns_with_usage
            else "No turn in this window reported token usage."
        ),
    )


def _render_confidence(summary: Summary) -> None:
    """How sure the system was, by the two measures that mean different things."""
    st.subheader("Confidence")
    left, right = st.columns(2)

    with left:
        st.caption(
            "**Retrieval** — how closely the best passage matched the question. "
            "Measured before the model writes, so a low score here is the earliest "
            "warning that an answer is about to be invented."
        )
        if not summary.turns_with_similarity:
            st.info(
                "No scored retrievals in this window. Turns recorded before "
                "similarity was tracked show nothing here rather than zero."
            )
        else:
            columns = st.columns(3)
            columns[0].metric(
                "Median top match",
                f"{summary.median_top_similarity:.2f}",
                help="Cosine similarity, where 1.00 is an identical vector.",
            )
            columns[1].metric(
                "Weak retrieval",
                _percent_text(summary.weak_retrieval_share),
                help=(
                    f"{summary.weak_retrieval_turns:,} of {summary.turns_with_similarity:,} "
                    f"turns found nothing above {WEAK_RETRIEVAL:.2f} — the corpus did not "
                    "really cover those questions."
                ),
            )
            # A high top match with no separation is the case the score alone hides:
            # everything matched equally, so the order the model read is arbitrary.
            columns[2].metric(
                "Match separation",
                f"{summary.shape.median_separation:.3f}",
                help=(
                    f"How far the best passage stood out from the rest of its own "
                    f"search. {summary.shape.flat_searches:,} of "
                    f"{summary.shape.searches_measured:,} searches separated by less "
                    f"than {FLAT_RETRIEVAL:.2f}, meaning nothing really won — the "
                    "embedding model is not telling these passages apart."
                ),
            )

    with right:
        st.caption(
            "**Faithfulness** — whether a second model finds the answer supported "
            "by the passages it cited. Says nothing about whether those passages "
            "are right, only whether the answer followed them."
        )
        if not summary.judged_turns:
            st.info("No turns judged yet — score some below.")
        else:
            columns = st.columns(2)
            columns[0].metric(
                "Mean faithfulness",
                f"{summary.mean_faithfulness:.2f}",
                help=f"Over the {summary.judged_turns:,} turns that have been judged.",
            )
            columns[1].metric(
                "Unsupported answers",
                _percent_text(summary.unfaithful_share),
                help=(
                    f"{summary.unfaithful_turns:,} of {summary.judged_turns:,} judged "
                    f"turns scored below {FAITHFUL_THRESHOLD:.2f}."
                ),
            )


def _render_outcomes(summary: Summary) -> None:
    """What the answers did, split three ways rather than graded on one bit.

    "Grounded" collapses two failures that want opposite fixes: an answer that
    declined is the system being honest about its corpus, and an answer that cited
    nothing without saying so is the system writing from memory. This is where they
    come apart.
    """
    refusal = summary.refusal
    st.subheader("What the answers did")

    columns = st.columns(4)
    columns[0].metric(
        "Grounded",
        _percent_text(summary.grounded_share),
        help=f"{summary.grounded_turns:,} of {summary.turns:,} answers cited a passage.",
    )
    columns[1].metric(
        "Declined",
        _percent_text(refusal.refusal_share),
        help=(
            f"{refusal.full:,} full and {refusal.partial:,} partial refusals, matched "
            "against the exact sentences the system prompt asks for."
        ),
    )
    columns[2].metric(
        "Unattributed",
        _percent_text(refusal.unattributed_share),
        help=(
            f"{refusal.unattributed:,} answers cited nothing and did not say why. "
            "Content with no source is the failure this page exists to surface."
        ),
    )
    columns[3].metric(
        "Refused without searching",
        f"{refusal.refused_without_searching:,}",
        help=(
            "Rule 1 of the system prompt requires a search before any factual "
            "answer. A refusal that ran none never gave the corpus a chance, so "
            "this needs no threshold to read as wrong."
        ),
    )

    if refusal.unattributed:
        st.warning(
            f"{refusal.unattributed:,} answer(s) cited none of the passages retrieved "
            "for them, and did not decline either. Content with no citation did not "
            "come from the documents."
        )
    if refusal.refused_without_searching:
        st.warning(
            f"{refusal.refused_without_searching:,} refusal(s) ran no search at all. "
            "The documents were never consulted before the answer said they did not "
            "cover the question."
        )

    st.caption(
        "**Was a refusal right?** Without labelled questions that cannot be "
        f"measured, so the split below is at {WEAK_RETRIEVAL:.2f} top match and sorts "
        "attention rather than scoring accuracy. Refusing over a strong match is a "
        "candidate false refusal; answering over a weak one is a candidate invention."
    )
    if not refusal.turns_with_similarity:
        st.info(
            "No turns in this window scored a passage, so there is nothing to split. "
            "A refusal that never searched has no match to have ignored — it is "
            "counted above instead."
        )
        return
    st.dataframe(
        [
            {
                "": "Declined",
                f"Strong match (≥ {WEAK_RETRIEVAL:.2f})": refusal.refused_on_strong_match,
                f"Weak match (< {WEAK_RETRIEVAL:.2f})": refusal.refused_on_weak_match,
            },
            {
                "": "Answered",
                f"Strong match (≥ {WEAK_RETRIEVAL:.2f})": refusal.answered_on_strong_match,
                f"Weak match (< {WEAK_RETRIEVAL:.2f})": refusal.answered_on_weak_match,
            },
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Over the {refusal.turns_with_similarity:,} turns that scored a passage. "
        "Top-left and bottom-right are the healthy cells."
    )


def _render_scoring(scoring: ScoreAnswers | None, window: int) -> None:
    """Judge answers on request, rather than while the reader waits for one."""
    if scoring is None:
        st.caption(
            "No judge model configured. Pick one under **Answer judge** on the "
            "setup screen (**Change models** in the sidebar) to score answers for "
            "faithfulness."
        )
        return

    pending = scoring.pending(window)
    if not pending:
        st.caption("Every recorded answer has been judged.")
        return

    waiting = (
        "1 answer has not been judged"
        if len(pending) == 1
        else f"{len(pending):,} answers have not been judged"
    )
    st.caption(
        f"{waiting}. Scoring calls the judge model once per answer, so it takes a "
        "while — the chat is untouched meanwhile."
    )
    # A slider whose ends meet is an error in Streamlit, and one pending answer
    # is not a choice worth offering anyway.
    most = min(len(pending), MAX_BATCH)
    batch = (
        st.slider("How many to score now", 1, most, min(most, DEFAULT_BATCH))
        if most > 1
        else 1
    )
    if not st.button(f"Score {batch} answer{'s' if batch > 1 else ''}", type="primary"):
        return

    progress = st.progress(0.0, text="Judging…")
    failures = []
    done = 0
    for turn, error in scoring.score(batch):
        done += 1
        progress.progress(done / batch, text=f"Judged {done} of {batch}…")
        if error:
            failures.append((turn.question, error))
    progress.empty()

    scored = done - len(failures)
    if scored:
        st.success(f"Scored {scored} of {done}.")
    for question, error in failures:
        st.warning(f"Could not judge “{_truncate(question, 60)}” — {error}")
    st.rerun()


def _render_trends(turns: Sequence[TurnRecord]) -> None:
    left, right = st.columns(2)
    with left:
        st.caption("**Answer time** — per turn, in the order they were answered")
        # Both series are durations in milliseconds, so they share one scale and
        # belong on one axis — two scales here would let the eye read a crossing
        # that is not in the data.
        st.line_chart(
            latency_series(turns),
            x=TURN_AXIS,
            y=[TOTAL_SERIES, FIRST_TOKEN_SERIES],
            height=260,
        )
    with right:
        st.caption("**Turns per day** (UTC)")
        st.bar_chart(per_day(turns), height=260)


def _render_documents(turns: Sequence[TurnRecord]) -> None:
    st.subheader("Retrieval by document")
    st.caption(
        "A document retrieved constantly and cited rarely is winning the search and "
        "losing the answer — which points at chunking, or at the missing reranker, "
        "rather than at the model."
    )
    rows = [
        {
            "Document": usage.source_file,
            "Retrieved": usage.retrieved,
            "Cited": usage.cited,
            "Cited share": _percent_number(usage.cited_share),
        }
        for usage in source_usage(turns)
    ]
    if not rows:
        st.caption("No passages retrieved in this window.")
        return
    st.dataframe(
        rows,
        hide_index=True,
        column_config={"Cited share": _share_column("Cited share")},
    )


def _render_searches(turns: Sequence[TurnRecord], summary: Summary) -> None:
    st.subheader("What the model searched for")
    st.caption(
        "The model rewrites, splits and narrows the question before searching, and a "
        "retrieval problem usually starts in that rewriting rather than in the index."
    )
    st.metric(
        "Searches returning nothing",
        _percent_text(summary.empty_search_share),
        help=f"{summary.empty_searches:,} of {summary.searches:,} searches came back empty.",
    )
    rows = [
        {"When": created_at, "Query": search.query, "Results": search.result_count}
        for created_at, search in recent_searches(turns, RECENT_SEARCHES)
    ]
    if rows:
        st.dataframe(rows, hide_index=True)


def _render_arms(turns: Sequence[TurnRecord], summary: Summary) -> None:
    """Which half of the hybrid search is doing the work.

    The fused order the model reads says nothing about where a passage came from,
    and that is the one thing a hybrid retriever is tuned on: an arm whose passages
    are never cited is adding noise, and an arm that only ever agrees with the
    other is not paying for itself.
    """
    arms = summary.arms
    st.subheader("Which arm found it")

    if not arms.attributed:
        st.info(
            f"None of the {arms.unattributed:,} retrieved passages in this window "
            "recorded which arm found them — they were logged before that was "
            "tracked. Ask something on the Chat page and it will fill in."
        )
        return

    columns = st.columns(4)
    columns[0].metric(
        "Keyword only",
        _percent_text(arms.keyword_only_share),
        help=f"{arms.keyword_only:,} passages BM25 found and the dense arm did not.",
    )
    columns[1].metric(
        "Dense only",
        _percent_text(arms.dense_only_share),
        help=f"{arms.dense_only:,} passages the embeddings found and BM25 did not.",
    )
    columns[2].metric(
        "Both arms",
        _percent_text(arms.overlap_share),
        help=(
            f"{arms.both:,} of {arms.attributed:,} attributed passages were found by "
            "both. Near zero means the arms agree on nothing and fusion is "
            "interleaving two unrelated opinions; near 100% means one arm is "
            "redundant."
        ),
    )
    columns[3].metric(
        "Arm not recorded",
        f"{arms.unattributed:,}",
        help=(
            "Passages logged before arm attribution existed. Not a finding, and "
            "unrelated to an unattributed *answer* above — that one is about "
            "citations, this one is about provenance."
        ),
    )

    st.dataframe(
        [
            {
                "Arm": row.arm,
                "Retrieved": row.retrieved,
                "Cited": row.cited,
                "Cited share": _percent_number(row.cited_share),
                "Median rank": row.median_final_rank,
            }
            for row in arm_usage(turns)
        ],
        hide_index=True,
        column_config={"Cited share": _share_column("Cited share")},
    )
    st.caption(
        "An arm retrieving plenty and cited rarely is winning the search and losing "
        "the answer. On a corpus the embedding model was not trained for, that is "
        "usually the dense arm — and the keyword arm quietly carrying every query."
    )


def _render_configurations(turns: Sequence[TurnRecord]) -> None:
    st.subheader("By configuration")
    st.caption(
        "Why the models are recorded at all: one configuration's numbers mean little "
        "until there is a second one to read them against."
    )
    st.dataframe(
        [
            {
                "Chat model": row.context.chat_model,
                "Embedding model": row.context.embed_model,
                "Reranker": row.context.reranker_model or "off",
                "Chunking": row.context.chunking_strategy,
                "Backend": row.context.vector_backend,
                "Turns": row.turns,
                "Median time": _duration(row.median_latency_ms),
                "Top match": round(row.median_top_similarity, 2) or None,
                "Grounded": _percent_number(row.grounded_share),
                "Faithfulness": (
                    round(row.mean_faithfulness, 2) if row.judged_turns else None
                ),
                "Passages / turn": round(row.passages_per_turn, 1),
            }
            for row in by_configuration(turns)
        ],
        hide_index=True,
        column_config={
            "Grounded": _share_column("Grounded"),
            "Top match": st.column_config.NumberColumn(
                "Top match", format="%.2f",
                help="Median best cosine similarity — whether this embedding model "
                     "finds closer matches for the same questions.",
            ),
            "Faithfulness": st.column_config.NumberColumn(
                "Faithfulness", format="%.2f",
                help="Blank where no turn under this configuration has been judged.",
            ),
        },
    )


def _render_turn(turn: TurnRecord) -> None:
    label = _truncate(turn.question, QUESTION_LABEL_CHARS) or "(empty question)"
    with st.expander(f"{turn.created_at} — {label}"):
        if turn.error:
            st.error(turn.error)
        st.markdown(turn.answer or "_No answer was produced._")
        facts = [
            _duration(turn.latency_ms),
            f"first token {_duration(turn.first_token_ms or 0)}",
            f"{turn.usage.total:,} tokens" if not turn.usage.is_empty else "tokens unreported",
            f"{len(turn.searches)} searches",
            f"{len(turn.retrievals)} retrieved",
            f"{turn.cited_count} cited",
        ]
        if turn.top_similarity is not None:
            facts.append(f"top match {turn.top_similarity:.2f}")
        state = outcome(turn)
        facts.append(f"**{state}**" if state == UNATTRIBUTED else state)
        if turn.refusal:
            facts.append(f"{turn.refusal} refusal")
        st.caption(" · ".join(facts))

        _render_judgement(turn)

        if turn.retrievals:
            st.dataframe(
                [
                    {
                        "Search": retrieval.search_index + 1,
                        "Rank": retrieval.rank,
                        "Arm": retrieval.arm,
                        "Fused": retrieval.fused_rank,
                        "Document": retrieval.source_file,
                        "Page": retrieval.page,
                        "Chunk": retrieval.chunk_index,
                        "Match": retrieval.score,
                        "Cited": retrieval.cited,
                    }
                    for retrieval in turn.retrievals
                ],
                hide_index=True,
                column_config={
                    # Blank rather than 0.00 where nothing measured it: these are
                    # the passages only the keyword arm found.
                    "Match": st.column_config.NumberColumn("Match", format="%.2f"),
                    # Where RRF ranked it, against `Rank` which is what the model
                    # finally read. They differ only when a reranker reordered them,
                    # and that difference is the reranker's whole contribution.
                    "Fused": st.column_config.NumberColumn(
                        "Fused",
                        help="Position after RRF, before any reranking.",
                    ),
                },
            )


def _render_judgement(turn: TurnRecord) -> None:
    judgement = turn.judgement
    if judgement is None:
        return
    line = f"**Faithfulness {judgement.faithfulness:.2f}** · judged by `{judgement.model}`"
    if judgement.is_faithful:
        st.success(line)
    else:
        st.warning(line)
    if judgement.unsupported:
        st.caption("Claims the judge could not find in the cited passages:")
        for claim in judgement.unsupported:
            st.caption(f"• {claim}")


# ── Page ──────────────────────────────────────────────────────────────────────

def render(statistics: ViewStatistics, scoring: ScoreAnswers | None = None) -> None:
    st.title("Statistics")

    window = st.selectbox(
        "Turns to read",
        WINDOW_OPTIONS,
        index=WINDOW_OPTIONS.index(DEFAULT_WINDOW),
        help="The most recent turns. Everything below is computed over these.",
    )
    # Deliberately uncached: the reason to open this page is usually the question
    # just asked, and a cached read is exactly the one that would not show it.
    turns = statistics.turns(window)

    if not turns:
        st.info("No turns recorded yet — ask something on the Chat page.")
        return

    summary = summarise(turns)
    _render_headline(summary)
    st.divider()
    _render_confidence(summary)
    st.divider()
    _render_outcomes(summary)
    _render_scoring(scoring, window)
    st.divider()
    _render_trends(turns)
    st.divider()
    _render_documents(turns)
    st.divider()
    _render_searches(turns, summary)
    st.divider()
    _render_arms(turns, summary)
    st.divider()
    _render_configurations(turns)
    st.divider()

    st.subheader("Recent turns")
    for turn in reversed(turns[-RECENT_TURNS:]):
        _render_turn(turn)
