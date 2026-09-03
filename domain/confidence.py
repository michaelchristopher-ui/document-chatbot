"""How far an answer stands up, worked out from the answer and nothing else.

Three readings, none of which costs a model call. That constraint is the whole
design: `application.judging` explains why a second model in the answer path is
not worth what it costs — minutes of a reader's wait, on a machine running a
large local model, for a number nobody is waiting to read. Everything here is
read off text and similarity scores that already exist by the time the answer
stops moving, so the score arrives with the answer rather than after it.

What each reading is worth knowing for:

- `retrieval` answers "did the corpus actually hold anything like this?", and it
  is the only one that would have been true before the model wrote a word.
- `citation_coverage` answers "is what it wrote anchored?" — the failure it
  catches is an answer that searched, found little, and wrote confidently anyway.
- `completeness` answers "did it address all of what was asked?" — the failure
  it catches is a part of a multi-part question dropped in silence, which the
  other two readings score perfectly.

The limit worth stating plainly: none of this reads the passages. An answer that
cites [1] after a claim [1] does not make scores well here whether or not the
passage says it. That is `Judgement`'s question, it needs a second model, and it
is asked later.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from domain.citations import cited_indices, prose, refusal_kind
from domain.constants import (
    ADDRESSED_OVERLAP,
    COMPLETENESS_WEIGHT,
    COVERAGE_WEIGHT,
    FULL_REFUSAL,
    HIGH,
    HIGH_CONFIDENCE,
    LOW,
    MEDIUM,
    MEDIUM_CONFIDENCE,
    MIN_CLAIM_WORDS,
    PARTIAL_REFUSAL,
    PARTIAL_REFUSAL_SENTENCE,
    REFUSAL_SENTENCE,
    RETRIEVAL_CEILING,
    RETRIEVAL_FLOOR,
    RETRIEVAL_WEIGHT,
    _PART_SPLIT,
    _SENTENCE_SPLIT,
    _STOPWORDS,
    _SUBPART_SPLIT,
    _WORD_RE,
)
from domain.models import AnswerConfidence, Citation


def assess(
    question: str, answer: str, citations: Sequence[Citation]
) -> AnswerConfidence:
    """Score one finished answer against the question and what it retrieved.

    Takes the answer whole rather than in pieces: two of the three readings are
    about what the finished text does — which of its claims carry markers, which
    parts of the question it got to — and neither is a fact about a fragment.

    A full refusal comes back scored None, carrying only how close the corpus
    came. Rule 7 tells the model to write exactly that sentence and cite nothing
    when the passages hold no answer, so every reading below would punish it for
    obeying: no claims to cite, no part addressed, and a low similarity that is
    the *evidence for* the refusal rather than a fault in it.
    """
    refusal = refusal_kind(answer)
    scores = [c.passage.score for c in citations if c.passage.score is not None]
    top = max(scores) if scores else None

    if refusal == FULL_REFUSAL:
        return AnswerConfidence(score=None, refusal=refusal, top_similarity=top)

    known = {citation.index for citation in citations}
    retrieval = _retrieval(citations, set(cited_indices(answer)) & known)

    claims = _claims(answer)
    cited = sum(1 for claim in claims if _is_anchored(claim, known))
    coverage = cited / len(claims) if claims else None

    vocabulary = _content(" ".join(prose(answer)))
    asked = _parts(question)
    reached = sum(1 for part in asked if _addresses(part, vocabulary))
    addressed, parts = _completeness_counts(reached, len(asked), refusal)
    completeness = addressed / parts if parts else None

    return AnswerConfidence(
        # Retrieval on its own is not a reading of the *answer*: it says the
        # corpus held something close to the question, which is as true of an
        # answer abandoned before its first token as of a good one. So an answer
        # that made no claim and reached no part of the question is unscored
        # rather than scored on the corpus's behalf — the components are still
        # reported, and the badge says it was not measured.
        score=(
            _composite(
                (retrieval, RETRIEVAL_WEIGHT),
                (coverage, COVERAGE_WEIGHT),
                (completeness, COMPLETENESS_WEIGHT),
            )
            if coverage is not None or completeness is not None
            else None
        ),
        retrieval=retrieval,
        citation_coverage=coverage,
        completeness=completeness,
        refusal=refusal,
        top_similarity=top,
        claims=len(claims),
        cited_claims=cited,
        parts=parts,
        addressed_parts=addressed,
    )


def band(confidence: AnswerConfidence) -> Optional[str]:
    """Which of three bands the composite falls in, or None when it has no score.

    A band rather than a number is what a reader acts on — open the sources, or
    do not — and the thresholds live here beside the weights that produce the
    number they cut, rather than in whichever UI happens to draw it.
    """
    if confidence.score is None:
        return None
    if confidence.score >= HIGH_CONFIDENCE:
        return HIGH
    if confidence.score >= MEDIUM_CONFIDENCE:
        return MEDIUM
    return LOW


def _retrieval(citations: Sequence[Citation], used: set[int]) -> Optional[float]:
    """How close the passages the answer leaned on were, on the 0–1 scale.

    Read over the cited passages rather than everything retrieved: a search
    returns five and an answer may need one, and the four it passed over say
    nothing about the one it used. Falls back to the whole result set for an
    answer that cited nothing, which is the only reading left for it.

    The best of them, not the average. A claim stands on one passage that says
    it, so citing a second, weaker passage beside a strong one is an answer being
    thorough — and an average would score it below the answer that cited only the
    strong one. `top_similarity` reports the same measure over everything
    retrieved, cited or not, which is the corpus's answer rather than this one's.

    None when no passage carries a score — the keyword arm found them all, and
    BM25 scores are on an unrelated scale. Unmeasured, not zero.
    """
    leaned = [c for c in citations if c.index in used] or list(citations)
    scores = [c.passage.score for c in leaned if c.passage.score is not None]
    if not scores:
        return None
    return _scale(max(scores))


def _scale(similarity: float) -> float:
    span = RETRIEVAL_CEILING - RETRIEVAL_FLOOR
    return min(1.0, max(0.0, (similarity - RETRIEVAL_FLOOR) / span))


def _claims(answer: str) -> list[str]:
    """The sentences of the answer that assert something worth a citation.

    Prose only, so a sentence inside a code fence is not a claim the answer is
    failing to cite. The prose spans are rejoined before the split rather than
    read one at a time: a term in backticks interrupts a sentence without ending
    it, and splitting per span would count "the notice period is `30d` in the
    contract [1]" as two claims and mark the first uncited. Rejoining on a space
    keeps every line break that was in the prose, so a fenced block still
    separates the paragraphs around it.

    The two mandated refusal sentences are not claims either: rules 7 and 8 tell
    the model to write them and cite nothing, and counting them would mark an
    answer down for following the instruction it was given.
    """
    sentences = _SENTENCE_SPLIT.split(" ".join(prose(answer)))
    return [
        sentence for sentence in (s.strip() for s in sentences) if _is_claim(sentence)
    ]


def _is_claim(sentence: str) -> bool:
    if REFUSAL_SENTENCE in sentence or PARTIAL_REFUSAL_SENTENCE in sentence:
        return False
    # Bare numbers dropped, which is how a citation marker's digits are kept from
    # counting as words of the sentence without a second copy of the marker
    # pattern here — `domain.citations` owns that pattern, and two that drifted
    # would quietly change what counts as a claim.
    words = [word for word in _WORD_RE.findall(sentence) if not word.isdigit()]
    return len(words) >= MIN_CLAIM_WORDS


def _is_anchored(claim: str, known: Iterable[int]) -> bool:
    """Whether the claim carries a marker pointing at a passage a search returned.

    An index nothing returned does not anchor it. The model is capable of writing
    `[7]` when six passages came back, and that is worth catching rather than
    counting: a marker the reader cannot follow is not a citation.
    """
    indices = set(known)
    return any(index in indices for index in cited_indices(claim))


def _parts(question: str) -> list[str]:
    """The question's distinct parts — what rule 9 tells the model to search for.

    Parts with no content words of their own are dropped rather than counted as
    unaddressed: "why?" after a first question is the reader ending a sentence,
    not a second thing to answer.
    """
    parts = (
        part.strip(" \t-•*")
        for sentence in _PART_SPLIT.split(question)
        for part in _SUBPART_SPLIT.split(sentence)
    )
    return [part for part in parts if _content(part)]


def _addresses(part: str, vocabulary: set[str]) -> bool:
    words = _content(part)
    return len(words & vocabulary) / len(words) >= ADDRESSED_OVERLAP


def _completeness_counts(
    reached: int, asked: int, refusal: Optional[str]
) -> tuple[int, int]:
    """How many parts the answer covered, out of how many, once rule 8 is read.

    A partial refusal names an aspect the documents do not cover, so the answer
    is incomplete by its own account — and its own account outranks word overlap,
    which reads the sentence naming the missing aspect as having addressed it. So
    one part is taken back: whatever the overlap found, the model said it did not
    answer all of this.

    That is also why a partial refusal to a question this reads as one part comes
    back as one of *two*. The model found a second aspect the splitter did not, and
    reporting "0 of 1" would say it answered nothing when it answered half.

    Counts rather than a share, because the reader is shown both — a badge saying
    "2 of 2 parts answered" beside a score that had already docked one would be
    the two halves of the same reading disagreeing on screen.
    """
    if refusal != PARTIAL_REFUSAL:
        return reached, asked
    parts = max(asked, 2)
    return min(reached, parts - 1), parts


def _composite(*readings: tuple[Optional[float], float]) -> Optional[float]:
    """The weighted mean of whatever was measured, over the weights that measured it.

    Renormalised rather than defaulted: a turn whose passages all came from the
    keyword arm has no retrieval reading, and folding that in as zero would
    report a weak search that never happened. Two readings out of three is a
    number worth showing; none is not.
    """
    measured = [(value, weight) for value, weight in readings if value is not None]
    if not measured:
        return None
    total = sum(weight for _, weight in measured)
    return sum(value * weight for value, weight in measured) / total


def _content(text: str) -> set[str]:
    """The topic-carrying words of `text`, crudely stemmed so plurals match."""
    return {
        _stem(word)
        for word in (w.lower() for w in _WORD_RE.findall(text))
        if len(word) >= 3 and word not in _STOPWORDS
    }


def _stem(word: str) -> str:
    """Enough stemming to match a plural to its singular, and no more.

    "Payments" in the question and "payment" in the answer are the same word for
    this purpose. A real stemmer would be a dependency and a language choice, and
    the reading this feeds is a soft one either way.
    """
    return word[:-1] if len(word) > 4 and word.endswith("s") else word
