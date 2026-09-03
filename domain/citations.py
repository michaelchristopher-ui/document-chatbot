from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, Iterator, Sequence

from domain.constants import (
    FULL_REFUSAL,
    NO_RESULTS_MESSAGE,
    PARTIAL_REFUSAL,
    PARTIAL_REFUSAL_SENTENCE,
    REFUSAL_SENTENCE,
    _CODE_RE,
    _MARKER_RE,
)
from domain.models import Citation, Passage


TitleLookup = Callable[[str], str]
RenderMarker = Callable[[Sequence[int]], str]


def passage_key(passage: Passage) -> tuple[str, int, str]:
    """What makes two retrieved spans the same passage.

    The text is part of it, not just the location: two chunking strategies split
    the same page differently, so a page and a filename alone would collapse
    spans that overlap without being the same span.

    Public because the ledger is not the only thing that has to decide whether it
    has seen a passage before — `domain.interactions` matches retrieved passages
    back to the numbers this ledger gave them, and two copies of the rule that
    drift would silently read every answer as citing nothing.
    """
    return (passage.source_file, passage.page, passage.text)


def header(citation: Citation) -> str:
    section = citation.passage.metadata.section
    return (
        f"Page {citation.passage.page} | {citation.title}"
        + (f" | {section}" if section else "")
    )


class CitationLedger:
    """Numbers passages once per answer, across every search that answer takes.

    A multi-part question triggers several searches and the model cites them all
    into one reply, so numbering that restarted at 1 for each search would leave
    `[1]` meaning two different passages in the same paragraph. A passage a later
    search returns again keeps the index it was first given.
    """

    def __init__(self, title_of: TitleLookup):
        self._title_of = title_of
        self._cited: dict[tuple[str, int, str], Citation] = {}

    def record(self, passages: Iterable[Passage]) -> tuple[Citation, ...]:
        """Cite `passages`, reusing the index of any already seen this answer."""
        cited = []
        for passage in passages:
            key = passage_key(passage)
            citation = self._cited.get(key)
            if citation is None:
                citation = Citation(
                    index=len(self._cited) + 1,
                    passage=passage,
                    title=self._title_of(passage.source_file),
                )
                self._cited[key] = citation
            cited.append(citation)
        return tuple(cited)


def format_citations(citations: Sequence[Citation]) -> str:
    """Render citations as the numbered context blocks the system prompt describes."""
    if not citations:
        return NO_RESULTS_MESSAGE
    return "\n\n".join(
        f"[{citation.index}] {header(citation)}\n{citation.passage.text}"
        for citation in citations
    )


def merge_citations(*groups: Iterable[Citation]) -> tuple[Citation, ...]:
    """Combine citation groups into one list in index order, without repeats."""
    merged: dict[int, Citation] = {}
    for group in groups:
        for citation in group:
            merged.setdefault(citation.index, citation)
    return tuple(citation for _, citation in sorted(merged.items()))


def cited_indices(answer: str) -> tuple[int, ...]:
    """The indices the answer cites, in order of first mention, without repeats.

    Citation-shaped literals inside code are not mentions.
    """
    seen: dict[int, None] = {}
    for span in prose(answer):
        for match in _MARKER_RE.finditer(span):
            for index in match.group(1).split(","):
                seen.setdefault(int(index), None)
    return tuple(seen)


def refusal_kind(answer: str) -> str | None:
    """Which of the two mandated refusals the answer makes, or None for neither.

    `FULL_REFUSAL` when the answer says the documents cover none of the question,
    `PARTIAL_REFUSAL` when it answers part of it and names what is missing. Full
    wins when both appear: an answer that has refused outright has refused.

    Matched against the sentences the system prompt instructs the model to write,
    read outside code fences by the same rule `cited_indices` uses — a refusal
    sentence quoted inside a fence is a quotation, not a refusal.

    What this separates: an answer that declined, and an answer that cited nothing
    and did not say why. `is_grounded` reads both as ungrounded, and only the
    second one is a failure.
    """
    spans = list(prose(answer))
    if any(REFUSAL_SENTENCE in span for span in spans):
        return FULL_REFUSAL
    if any(PARTIAL_REFUSAL_SENTENCE in span for span in spans):
        return PARTIAL_REFUSAL
    return None


def displayed_citations(
    answer: str, citations: Iterable[Citation]
) -> dict[int, Citation]:
    """Each index the answer cites, mapped to that passage under its footnote number.

    Two things happen here, both presentational. Only cited passages survive: a
    search casts wide on purpose, and listing what the answer never leaned on
    credits it to passages it did not use. What survives is then renumbered from
    one in order of first mention, because the ledger numbers passages as
    searches return them — an answer built from the first and fourth would
    otherwise carry `[1]` and `[4]` above a list of two, reading as though two
    sources had gone missing.

    Keys are the indices the model wrote, so markers in the answer can be looked
    up; each value carries the number the reader should see in its place.
    """
    by_index = {citation.index: citation for citation in citations}
    cited = (index for index in cited_indices(answer) if index in by_index)
    return {
        index: replace(by_index[index], index=number)
        for number, index in enumerate(cited, 1)
    }


def rewrite_markers(answer: str, render: RenderMarker) -> str:
    """Replace every citation marker outside code with `render(indices)`.

    A presentation seam: the domain knows which text is a citation marker, and
    the caller decides what a reader should see in its place.
    """
    return "".join(
        _rewrite(span, render) if is_prose else span
        for span, is_prose in _segments(answer)
    )


def _segments(text: str) -> Iterator[tuple[str, bool]]:
    """`text` in order, as (span, is_prose) pairs that rejoin into the original."""
    cursor = 0
    for code in _CODE_RE.finditer(text):
        yield text[cursor:code.start()], True
        yield code.group(), False
        cursor = code.end()
    yield text[cursor:], True


def prose(text: str) -> Iterator[str]:
    """The spans of `text` that are not code, in order.

    Public because reading an answer is not only this module's business any more:
    `domain.confidence` counts the claims an answer makes, and a claim inside a
    code fence is no more a claim than a citation marker there is a citation. One
    rule for what counts as prose, applied by everything that reads an answer.
    """
    return (span for span, is_prose in _segments(text) if is_prose)


def _rewrite(text: str, render: RenderMarker) -> str:
    return _MARKER_RE.sub(
        lambda match: render([int(index) for index in match.group(1).split(",")]),
        text,
    )
