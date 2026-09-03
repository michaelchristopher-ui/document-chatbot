"""What one answered turn is worth remembering, and how that is worked out.

Two shapes of the same turn live here. `SearchOutcome` is the live capture, taken
as the answer happens: whole passages, straight off the retriever. `TurnRecord`
and its parts are what is stored and read back, and they hold no passage text —
only where a passage came from and whether the answer leaned on it. The corpus
already keeps the text, and a log that kept it too would grow like a second copy
of it for no question this answers.

`build_turn` is the seam between the two, and the only place that decides what
"grounded" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from domain.citations import cited_indices, passage_key, refusal_kind
from domain.constants import (
    ARM_BOTH,
    ARM_DENSE,
    ARM_KEYWORD,
    ARM_UNATTRIBUTED,
    FAITHFUL_THRESHOLD,
)
from domain.models import Citation, Passage, RetrievalOrigin


@dataclass(frozen=True)
class TurnContext:
    """The configuration a turn was answered under.

    Frozen and all-strings so turns group by it directly: the per-configuration
    comparison is a `dict` keyed on this value. Which is the point of recording
    the models at all — one configuration's numbers mean little until there is a
    second one to read them against.
    """

    chat_model: str
    embed_model: str
    reranker_model: str
    chunking_strategy: str
    vector_backend: str


@dataclass(frozen=True)
class SearchOutcome:
    """One `search_documents` call as it happened, passages and all."""

    query: str
    passages: tuple[Passage, ...] = ()


@dataclass(frozen=True)
class SearchRecord:
    """A search as it is stored: what was asked, and how much came back."""

    index: int
    query: str
    result_count: int


@dataclass(frozen=True)
class RetrievalRecord:
    """One passage a search returned, and whether the answer cited it.

    `rank` is 1-based within its own search, and it is the only ordering signal
    that survives: passages carry no similarity score, so this is the position
    the answer actually saw the passage in.

    `citation_index` is the number the ledger gave the passage — the number the
    model writes. None means the passage never reached the ledger, which should
    not happen; it is recorded rather than raised, because a turn that has
    already been answered should still be logged.

    `score` is the cosine similarity the search matched it at, or None for a
    passage the keyword arm found and the dense arm did not — BM25 scores are on
    an unrelated scale, and averaging the two would produce a number that means
    nothing.
    """

    search_index: int
    rank: int
    source_file: str
    page: int
    chunk_index: int
    citation_index: Optional[int]
    cited: bool
    score: Optional[float] = None
    # The passage as the model read it. Carried here because a judge scoring the
    # answer later has no other way back to it — every ingest rebuilds the vector
    # store, so the chunk this came from may not exist by then.
    text: Optional[str] = None
    # Where each arm of the hybrid search had this passage before fusion, and where
    # fusion put it. Flat rather than nested because this is the stored shape and
    # each field is one column; `domain.models.RetrievalOrigin` is the live one.
    # All None for a row written before any of this was recorded.
    keyword_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    fused_rank: Optional[int] = None

    @property
    def arm(self) -> str:
        """Which arm of the search found this passage.

        ARM_UNATTRIBUTED when neither rank was recorded — the row predates arm
        attribution, or came from a path that could not attribute it. Distinct from
        a passage one arm missed, which has that arm's rank as None while the other
        carries a number.
        """
        if self.keyword_rank is not None and self.dense_rank is not None:
            return ARM_BOTH
        if self.keyword_rank is not None:
            return ARM_KEYWORD
        if self.dense_rank is not None:
            return ARM_DENSE
        return ARM_UNATTRIBUTED


@dataclass(frozen=True)
class TokenUsage:
    """Tokens spent across every model call in a turn.

    Summed rather than replaced, since the ReAct loop calls the model once per
    search plus once more to answer.
    """

    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    @property
    def is_empty(self) -> bool:
        """True when nothing was reported — a backend that stays quiet about usage."""
        return self.total == 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(self.prompt + other.prompt, self.completion + other.completion)


@dataclass(frozen=True)
class Judgement:
    """A second model's reading of whether an answer follows from its sources.

    Kept apart from the turn because it is made later, by a different model, and
    a turn can be judged long after it was answered — or never. `faithfulness`
    runs 0.0 to 1.0; `unsupported` names the claims the judge could not find in
    the passages, which is the part worth reading when the score is low.

    This measures whether the answer follows from what was retrieved, not whether
    it is true. An answer faithful to a passage that is itself wrong scores 1.0.
    """

    faithfulness: float
    unsupported: tuple[str, ...]
    model: str
    judged_at: str

    @property
    def is_faithful(self) -> bool:
        return self.faithfulness >= FAITHFUL_THRESHOLD


@dataclass(frozen=True)
class TurnRecord:
    """One question and its answer, with what the answer used and what it cost.

    `first_token_ms` is time to the first token *of the answer*, so it counts the
    searches the model made before writing anything — which is the wait a reader
    actually sits through. A model that writes a preamble before searching moves
    this number earlier; that is the preamble, not a faster answer.

    None rather than zero wherever a measurement is missing: a turn abandoned
    before its first token, and a backend that reports no usage, are both normal,
    and a zero would average into the statistics as though it were measured.
    """

    thread_id: str
    created_at: str
    question: str
    answer: str
    latency_ms: int
    first_token_ms: Optional[int]
    usage: TokenUsage
    context: TurnContext
    searches: tuple[SearchRecord, ...] = ()
    retrievals: tuple[RetrievalRecord, ...] = ()
    error: Optional[str] = None
    # Assigned by the log on the way out, so a judgement made later can name the
    # turn it is about. None for a turn that has not been stored yet.
    id: Optional[int] = None
    judgement: Optional[Judgement] = None

    @property
    def cited_count(self) -> int:
        return sum(1 for retrieval in self.retrievals if retrieval.cited)

    @property
    def refusal(self) -> Optional[str]:
        """Which mandated refusal this answer makes, if any — see `refusal_kind`.

        Derived rather than stored, like every other reading here: the answer is
        kept in full, and a stored copy would be free to disagree with it. It also
        means every turn already logged gains the reading retroactively.
        """
        return refusal_kind(self.answer)

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(r.score for r in self.retrievals if r.score is not None)

    @property
    def top_similarity(self) -> Optional[float]:
        """How well the closest passage matched — the corpus's answer to the question.

        Derived rather than stored, like every other count here: the passages are
        loaded anyway, and a copy on the turn would be free to disagree with them.

        This is the cheapest warning that an answer is about to be invented. A
        question the documents simply do not cover scores low here *before* the
        model writes a word, whereas groundedness only reveals it afterwards.
        """
        return max(self.scores) if self.scores else None

    @property
    def mean_similarity(self) -> Optional[float]:
        return sum(self.scores) / len(self.scores) if self.scores else None


def build_turn(
    *,
    thread_id: str,
    created_at: str,
    question: str,
    answer: str,
    latency_ms: int,
    first_token_ms: Optional[int],
    usage: TokenUsage,
    context: TurnContext,
    outcomes: Sequence[SearchOutcome],
    citations: Sequence[Citation],
    error: Optional[str] = None,
) -> TurnRecord:
    """Fold a turn's live capture into the row it is stored as.

    Grounding is settled here, from two things that already exist: the ledger
    numbered every retrieved passage, and `cited_indices` reads which of those
    numbers the answer went on to write. A citation-shaped literal inside a code
    fence is not a mention, because that function already says so.

    One row per passage *per search*, not per turn. A passage two searches both
    return earns two rows, which is what makes "did the second search find
    anything the first did not?" a question the log can answer; collapsing to the
    turn is a group-by away, and the reverse is not.
    """
    numbering = {passage_key(citation.passage): citation.index for citation in citations}
    written = set(cited_indices(answer))

    searches = []
    retrievals = []
    for search_index, outcome in enumerate(outcomes):
        searches.append(
            SearchRecord(
                index=search_index,
                query=outcome.query,
                result_count=len(outcome.passages),
            )
        )
        for rank, passage in enumerate(outcome.passages, start=1):
            index = numbering.get(passage_key(passage))
            # A passage with no origin keeps three Nones rather than being skipped:
            # it was still retrieved, and `RetrievalRecord.arm` reads that as
            # unattributed rather than as an arm that found nothing.
            origin = passage.origin or RetrievalOrigin()
            retrievals.append(
                RetrievalRecord(
                    search_index=search_index,
                    rank=rank,
                    source_file=passage.source_file,
                    page=passage.page,
                    chunk_index=passage.metadata.index,
                    citation_index=index,
                    cited=index is not None and index in written,
                    score=passage.score,
                    text=passage.text,
                    keyword_rank=origin.keyword_rank,
                    dense_rank=origin.dense_rank,
                    fused_rank=origin.fused_rank,
                )
            )

    return TurnRecord(
        thread_id=thread_id,
        created_at=created_at,
        question=question,
        answer=answer,
        latency_ms=latency_ms,
        first_token_ms=first_token_ms,
        usage=usage,
        context=context,
        searches=tuple(searches),
        retrievals=tuple(retrievals),
        error=error,
    )
