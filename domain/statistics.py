"""Reading a pile of recorded turns as numbers worth acting on.

Arithmetic rather than SQL, for the same reason `domain.fusion` and
`domain.dedup` are: it belongs where it can be read and exercised without a
database. Two of these are not expressible in SQLite anyway — it has no median
and no percentile — and groundedness leans on `domain.citations.cited_indices`,
whose rule about citation markers inside code fences would have to be written a
second time in SQL and would drift from the first.

The cost is that the caller loads a window of turns into memory. That is what the
`limit` on the read port is for.

Every ratio here answers 0.0 for an empty window: a log with nothing in it is the
normal first run, not an error.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from domain.constants import (
    ARM_BOTH,
    ARM_DENSE,
    ARM_KEYWORD,
    ARM_UNATTRIBUTED,
    FIRST_TOKEN_SERIES,
    FLAT_RETRIEVAL,
    FULL_REFUSAL,
    GROUNDED,
    MEDIAN,
    P95,
    PARTIAL_REFUSAL,
    REFUSED,
    TOTAL_SERIES,
    TURN_AXIS,
    UNATTRIBUTED,
    WEAK_RETRIEVAL,
)
from domain.interactions import SearchRecord, TokenUsage, TurnContext, TurnRecord


def _median_float(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _share(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def percentile(values: Sequence[int], fraction: float) -> int:
    """Nearest-rank percentile — an observed value, never an interpolated one.

    0 for an empty sequence, which callers show as a dash rather than a reading.
    """
    if not values:
        return 0
    ordered = sorted(values)
    rank = math.ceil(fraction * len(ordered))
    return ordered[min(len(ordered) - 1, max(0, rank - 1))]


def is_grounded(turn: TurnRecord) -> bool:
    """True when the answer cited at least one passage a search returned.

    The headline measure of whether retrieval is doing anything: an answer that
    cites nothing was either refused or written from the model's own memory, and
    both are failures of the same thing.
    """
    return any(retrieval.cited for retrieval in turn.retrievals)


def outcome(turn: TurnRecord) -> str:
    """Which of the three things an answer did, as one word.

    `is_grounded` collapses the last two: an answer that declined and an answer
    that cited nothing without saying so both read as ungrounded, and only the
    second is a failure. Splitting them is the difference between a system that is
    honest about its corpus and one that is writing from memory.

    A partial refusal lands in GROUNDED when it cited the part it could answer,
    which is the behaviour rule 8 asks for; `RefusalTotals` counts it separately.
    """
    if is_grounded(turn):
        return GROUNDED
    if turn.refusal:
        return REFUSED
    return UNATTRIBUTED


@dataclass(frozen=True)
class ScoreShape:
    """How far the best match stood out from the rest of one ranking.

    `measured` is how many of that ranking's passages carried a score at all, so a
    shape drawn from two of five passages can be told from one drawn from all five.
    """

    top: float
    mean_rest: float
    measured: int

    @property
    def separation(self) -> float:
        return self.top - self.mean_rest

    @property
    def is_flat(self) -> bool:
        return self.separation < FLAT_RETRIEVAL


def score_shape(scores: Sequence[Optional[float]]) -> Optional[ScoreShape]:
    """The shape of one ranking's scores, or None when fewer than two were measured.

    Nones are dropped, never zero-filled: a passage the keyword arm found and the
    dense arm did not is unmeasured, and a zero there would read as a perfect
    non-match and invent a separation that is not in the data.
    """
    measured = [score for score in scores if score is not None]
    if len(measured) < 2:
        return None
    # Everything but one copy of the best, so a tie at the top does not remove both.
    rest = sorted(measured)[:-1]
    return ScoreShape(
        top=max(measured),
        mean_rest=sum(rest) / len(rest),
        measured=len(measured),
    )


def search_shapes(turns: Sequence[TurnRecord]) -> list[ScoreShape]:
    """One shape per search that scored at least two passages, oldest first.

    Grouped by search rather than by turn: two searches in one turn ask two
    different questions, and pooling their scores measures neither.
    """
    shapes = []
    for turn in turns:
        by_search: dict[int, list[Optional[float]]] = defaultdict(list)
        for retrieval in turn.retrievals:
            by_search[retrieval.search_index].append(retrieval.score)
        for _, scores in sorted(by_search.items()):
            shape = score_shape(scores)
            if shape is not None:
                shapes.append(shape)
    return shapes


@dataclass(frozen=True)
class ShapeTotals:
    """How much the best match stood out, across every search that scored two."""

    searches_measured: int
    median_separation: float
    flat_searches: int

    @property
    def flat_share(self) -> float:
        """Of the searches that were measured, how many found nothing that stood out."""
        return _share(self.flat_searches, self.searches_measured)


@dataclass(frozen=True)
class RefusalTotals:
    """Refusals and unattributed answers, against what retrieval actually found.

    Detecting a refusal is only half of it. The half that matters is whether the
    refusal was right, and without labelled questions that cannot be measured — so
    what is offered instead is two readings that need no labels.

    `refused_without_searching` is the sharp one: rule 1 of the system prompt
    requires a search before any factual answer, so a refusal that ran none is a
    compliance failure and needs no threshold to interpret.

    The four quadrants are the soft one. They split at `WEAK_RETRIEVAL`, so they
    sort attention rather than measure accuracy: a refusal over a strong match is a
    candidate false refusal, and an answer over a weak one is the candidate
    invention. Labelled questions are what would turn either into a number.
    """

    turns: int
    full: int
    partial: int
    unattributed: int
    refused_without_searching: int
    refused_on_strong_match: int
    refused_on_weak_match: int
    answered_on_weak_match: int
    answered_on_strong_match: int
    turns_with_similarity: int

    @property
    def refusals(self) -> int:
        return self.full + self.partial

    @property
    def refusal_share(self) -> float:
        return _share(self.refusals, self.turns)

    @property
    def unattributed_share(self) -> float:
        """Answers that cited nothing and did not say why — the quiet failure."""
        return _share(self.unattributed, self.turns)

    @property
    def suspect_refusal_share(self) -> float:
        """Of the refusals, how many look wrong: no search run, or a strong match ignored."""
        return _share(
            self.refused_without_searching + self.refused_on_strong_match,
            self.refusals,
        )


@dataclass(frozen=True)
class Summary:
    turns: int
    grounded_turns: int
    median_latency_ms: int
    p95_latency_ms: int
    median_first_token_ms: int
    searches: int
    empty_searches: int
    retrieved: int
    cited: int
    usage: TokenUsage
    turns_with_usage: int
    # Retrieval confidence — how well the corpus matched the questions asked.
    median_top_similarity: float
    weak_retrieval_turns: int
    turns_with_similarity: int
    # Judged faithfulness — only over the turns a judge has actually read.
    judged_turns: int
    mean_faithfulness: float
    unfaithful_turns: int
    # How well the best match stood out, and what the answers did with it.
    shape: ShapeTotals
    refusal: RefusalTotals
    # Which arm of the hybrid search found what.
    arms: ArmTotals

    @property
    def weak_retrieval_share(self) -> float:
        """Of the turns that were scored, how many found nothing that matched well."""
        return _share(self.weak_retrieval_turns, self.turns_with_similarity)

    @property
    def unfaithful_share(self) -> float:
        return _share(self.unfaithful_turns, self.judged_turns)

    @property
    def grounded_share(self) -> float:
        return _share(self.grounded_turns, self.turns)

    @property
    def cited_share(self) -> float:
        """How much of what retrieval returned the answers actually used."""
        return _share(self.cited, self.retrieved)

    @property
    def empty_search_share(self) -> float:
        return _share(self.empty_searches, self.searches)

    @property
    def searches_per_turn(self) -> float:
        return _share(self.searches, self.turns)

    @property
    def passages_per_turn(self) -> float:
        return _share(self.retrieved, self.turns)


def summarise(turns: Sequence[TurnRecord]) -> Summary:
    latencies = [turn.latency_ms for turn in turns]
    # A turn abandoned before it wrote anything has no time-to-first-token. It is
    # dropped rather than counted as zero, which would drag the median toward a
    # speed nothing ever achieved.
    first_tokens = [
        turn.first_token_ms for turn in turns if turn.first_token_ms is not None
    ]
    searches = [search for turn in turns for search in turn.searches]
    retrievals = [retrieval for turn in turns for retrieval in turn.retrievals]

    usage = TokenUsage()
    measured = 0
    for turn in turns:
        if not turn.usage.is_empty:
            usage = usage + turn.usage
            measured += 1

    # Only turns whose search actually scored something. A turn recorded before
    # scores existed, or answered without searching, is absent rather than zero.
    similarities = [t.top_similarity for t in turns if t.top_similarity is not None]
    judged = [t.judgement for t in turns if t.judgement is not None]

    shapes = search_shapes(turns)
    flat = sum(1 for shape in shapes if shape.is_flat)

    full = partial = unattributed = unsearched = 0
    # Keyed (refused, strong_match) so the four quadrants are one pass.
    quadrants: Counter = Counter()
    for turn in turns:
        refused = turn.refusal is not None
        full += turn.refusal == FULL_REFUSAL
        partial += turn.refusal == PARTIAL_REFUSAL
        unattributed += outcome(turn) == UNATTRIBUTED
        # A refusal that never searched, which rule 1 forbids outright. Counted
        # over refusals only: an *answer* without a search is a different fault.
        unsearched += refused and not turn.searches
        if turn.top_similarity is not None:
            quadrants[(refused, turn.top_similarity >= WEAK_RETRIEVAL)] += 1

    arms: Counter = Counter(
        retrieval.arm for turn in turns for retrieval in turn.retrievals
    )
    return Summary(
        arms=ArmTotals(
            keyword_only=arms[ARM_KEYWORD],
            dense_only=arms[ARM_DENSE],
            both=arms[ARM_BOTH],
            attributed=arms[ARM_KEYWORD] + arms[ARM_DENSE] + arms[ARM_BOTH],
            unattributed=arms[ARM_UNATTRIBUTED],
        ),
        shape=ShapeTotals(
            searches_measured=len(shapes),
            median_separation=_median_float([s.separation for s in shapes]),
            flat_searches=flat,
        ),
        refusal=RefusalTotals(
            turns=len(turns),
            full=full,
            partial=partial,
            unattributed=unattributed,
            refused_without_searching=unsearched,
            refused_on_strong_match=quadrants[(True, True)],
            refused_on_weak_match=quadrants[(True, False)],
            answered_on_weak_match=quadrants[(False, False)],
            answered_on_strong_match=quadrants[(False, True)],
            turns_with_similarity=len(similarities),
        ),
        median_top_similarity=_median_float(similarities),
        weak_retrieval_turns=sum(1 for s in similarities if s < WEAK_RETRIEVAL),
        turns_with_similarity=len(similarities),
        judged_turns=len(judged),
        mean_faithfulness=(
            sum(j.faithfulness for j in judged) / len(judged) if judged else 0.0
        ),
        unfaithful_turns=sum(1 for j in judged if not j.is_faithful),
        turns=len(turns),
        grounded_turns=sum(1 for turn in turns if is_grounded(turn)),
        median_latency_ms=percentile(latencies, MEDIAN),
        p95_latency_ms=percentile(latencies, P95),
        median_first_token_ms=percentile(first_tokens, MEDIAN),
        searches=len(searches),
        empty_searches=sum(1 for search in searches if search.result_count == 0),
        retrieved=len(retrievals),
        cited=sum(1 for retrieval in retrievals if retrieval.cited),
        usage=usage,
        turns_with_usage=measured,
    )


def per_day(turns: Sequence[TurnRecord]) -> dict[str, int]:
    """Turns per calendar day (UTC), oldest first."""
    counts = Counter(turn.created_at[:10] for turn in turns)
    return {day: counts[day] for day in sorted(counts)}


def latency_series(turns: Sequence[TurnRecord]) -> list[dict[str, int]]:
    """Both latency measures per turn, in the order they were answered.

    Rows rather than columns, each carrying a 1-based ordinal, because a plot
    given only the two series has to invent an x axis and numbers its points
    0.0, 0.5, 1.0 — a scale that means nothing for something counted in turns.

    Time to first token is carried forward across a turn that never produced one,
    so a gap reads as unchanged rather than as an instant answer.
    """
    rows = []
    carried = 0
    for ordinal, turn in enumerate(turns, start=1):
        carried = turn.first_token_ms if turn.first_token_ms is not None else carried
        rows.append(
            {
                TURN_AXIS: ordinal,
                TOTAL_SERIES: turn.latency_ms,
                FIRST_TOKEN_SERIES: carried,
            }
        )
    return rows


@dataclass(frozen=True)
class SourceUsage:
    """How often one document is found, against how often it is used.

    A document retrieved constantly and cited never is the signal worth chasing:
    it is winning the search and losing the answer, which points at chunking or
    at the absence of a reranker rather than at the model.
    """

    source_file: str
    retrieved: int
    cited: int

    @property
    def cited_share(self) -> float:
        return _share(self.cited, self.retrieved)


def source_usage(turns: Sequence[TurnRecord]) -> list[SourceUsage]:
    """Per document, most-retrieved first."""
    retrieved: Counter = Counter()
    cited: Counter = Counter()
    for turn in turns:
        for retrieval in turn.retrievals:
            retrieved[retrieval.source_file] += 1
            if retrieval.cited:
                cited[retrieval.source_file] += 1
    return [
        SourceUsage(source_file=name, retrieved=count, cited=cited[name])
        for name, count in retrieved.most_common()
    ]


@dataclass(frozen=True)
class ArmUsage:
    """How much one arm of the hybrid search contributed, and how much was used.

    What RRF hides: a keyword arm whose passages are never cited is adding noise
    the fusion then ranks, and a dense arm contributing nothing the keyword arm
    missed is not earning its embedding call. Neither is visible in the fused order.
    """

    arm: str
    retrieved: int
    cited: int
    median_final_rank: int

    @property
    def cited_share(self) -> float:
        return _share(self.cited, self.retrieved)


def arm_usage(turns: Sequence[TurnRecord]) -> list[ArmUsage]:
    """Per arm, most-retrieved first.

    ARM_UNATTRIBUTED gets its own row rather than being folded in: those passages
    were not measured, and adding them to an arm would claim a reading nobody took.
    """
    retrieved: Counter = Counter()
    cited: Counter = Counter()
    ranks: dict[str, list[int]] = defaultdict(list)
    for turn in turns:
        for retrieval in turn.retrievals:
            retrieved[retrieval.arm] += 1
            ranks[retrieval.arm].append(retrieval.rank)
            if retrieval.cited:
                cited[retrieval.arm] += 1
    return [
        ArmUsage(
            arm=arm,
            retrieved=count,
            cited=cited[arm],
            median_final_rank=percentile(ranks[arm], MEDIAN),
        )
        for arm, count in retrieved.most_common()
    ]


@dataclass(frozen=True)
class ArmTotals:
    """How often the two arms agreed, over the passages that were attributed.

    Overlap is the number worth reading. Near zero means the arms are finding
    disjoint sets and RRF is interleaving two opinions with nothing in common;
    near one means they agree and one of them could be switched off.
    """

    keyword_only: int
    dense_only: int
    both: int
    attributed: int
    unattributed: int

    @property
    def overlap_share(self) -> float:
        return _share(self.both, self.attributed)

    @property
    def keyword_only_share(self) -> float:
        return _share(self.keyword_only, self.attributed)

    @property
    def dense_only_share(self) -> float:
        return _share(self.dense_only, self.attributed)


@dataclass(frozen=True)
class ConfigurationUsage:
    context: TurnContext
    turns: int
    median_latency_ms: int
    grounded_share: float
    passages_per_turn: float
    # The number that actually compares two embedding models: whether one finds
    # closer matches for the same questions.
    median_top_similarity: float
    mean_faithfulness: float
    judged_turns: int


def by_configuration(turns: Sequence[TurnRecord]) -> list[ConfigurationUsage]:
    """One row per configuration answered under, busiest first."""
    grouped: dict[TurnContext, list[TurnRecord]] = defaultdict(list)
    for turn in turns:
        grouped[turn.context].append(turn)

    rows = []
    for context, group in grouped.items():
        summary = summarise(group)
        rows.append(
            ConfigurationUsage(
                context=context,
                turns=summary.turns,
                median_latency_ms=summary.median_latency_ms,
                grounded_share=summary.grounded_share,
                passages_per_turn=summary.passages_per_turn,
                median_top_similarity=summary.median_top_similarity,
                mean_faithfulness=summary.mean_faithfulness,
                judged_turns=summary.judged_turns,
            )
        )
    return sorted(rows, key=lambda row: row.turns, reverse=True)


def recent_searches(
    turns: Sequence[TurnRecord], limit: int
) -> list[tuple[str, SearchRecord]]:
    """The searches the model chose to run, newest first, paired with their time.

    What the model looked for is not what the user asked — it rewrites, splits
    and narrows — and that rewriting is where a retrieval problem usually starts.
    """
    pairs = [
        (turn.created_at, search) for turn in turns for search in turn.searches
    ]
    return list(reversed(pairs))[:limit]
