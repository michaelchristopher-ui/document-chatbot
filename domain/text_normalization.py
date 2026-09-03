"""Page-text normalization, as a chain of steps that can be added or removed.

What a PDF extractor hands back carries the furniture of the printed page:
running heads, page numbers, a hyphen wherever a word broke at the right margin,
and one hard line break per printed line. None of it is content, and all of it
reaches the index — BM25 tokenises `individ-` and `ﬁnd` as terms of their own,
`domain.chunking` reads a running head as a section heading, and a header
repeated on every page pushes otherwise distinct chunks toward the near-duplicate
threshold in `domain.dedup`.

Every step has the same shape — `Sequence[PageText] -> list[PageText]` — so a
chain is assembled, reordered or cut down wherever it is built:

    build_normalizer()                                  # DEFAULT_STEPS
    build_normalizer(DEFAULT_STEPS + (straighten_quotes,))
    build_normalizer(s for s in DEFAULT_STEPS if s is not join_wrapped_lines)

A step sees the whole document rather than one page because two of them have to:
a running head is only recognisable as one by repeating across pages, and a page
number only by counting in step with them. The rest are page-local and say so.

Nothing in `DEFAULT_STEPS` changes what a reader sees on the page — text is
either furniture being removed or a word being put back together across the line
break that split it. `straighten_quotes` is the one step that does alter
characters a reader would notice, which is why it is offered but not enabled.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import replace
from math import ceil
from typing import Callable, Iterable, Sequence

from domain.constants import (
    EDGE_LINES,
    MIN_PAGES_FOR_DETECTION,
    RUNNING_HEAD_MAX_LEN,
    RUNNING_HEAD_MIN_PAGES,
    RUNNING_HEAD_RATIO,
    WIDTH_PERCENTILE,
    WRAP_MIN_WIDTH,
    WRAP_WIDTH_RATIO,
    _CHARACTER_MAP,
    _HYPHEN_BREAK_RE,
    _PAGE_NUMBER_RE,
    _QUOTE_MAP,
)
from domain.models import PageText

Step = Callable[[Sequence[PageText]], list[PageText]]
Normalizer = Callable[[Sequence[PageText]], list[PageText]]


# ── The chain ─────────────────────────────────────────────────────────────────

def build_normalizer(steps: Iterable[Step] | None = None) -> Normalizer:
    """Fold `steps` into one callable, applied in order. Defaults to DEFAULT_STEPS."""
    chain = tuple(DEFAULT_STEPS if steps is None else steps)

    def normalize(pages: Sequence[PageText]) -> list[PageText]:
        result = list(pages)
        for step in chain:
            result = step(result)
        return result

    return normalize


def _map_pages(pages: Sequence[PageText], transform: Callable[[str], str]) -> list[PageText]:
    """Apply a page-local transform, dropping pages it empties."""
    rewritten = (replace(page, text=transform(page.text)) for page in pages)
    return [page for page in rewritten if page.text.strip()]


# ── Steps ─────────────────────────────────────────────────────────────────────

def normalize_characters(pages: Sequence[PageText]) -> list[PageText]:
    """Page-local. Resolve ligatures, dashes and invisibles to what is written.

    Deliberately not NFKC, which resolves ligatures but folds superscripts into
    digits with them, turning `R²` into `R2` and `m³` into `m3`. Ligatures are
    mapped by name instead, and NFC then recomposes accents so that a word is
    spelled one way whichever route its producer took to set it.
    """
    return _map_pages(
        pages, lambda text: unicodedata.normalize("NFC", text.translate(_CHARACTER_MAP))
    )


def drop_running_heads(pages: Sequence[PageText]) -> list[PageText]:
    """Document-wide. Remove the title lines that repeat at the top or foot.

    Counted once per page, so a head that also appears in the body is judged on
    how many pages carry it rather than how often it is set.

    The first page is counted but never stripped: what a running head repeats is
    usually the title, and the first page is where that title is set as content
    rather than as furniture — the one place the document says its own name.
    """
    if len(pages) < MIN_PAGES_FOR_DETECTION:
        return list(pages)

    seen: Counter[str] = Counter()
    for page in pages:
        lines = page.text.split("\n")
        seen.update(
            {
                lines[index].strip()
                for index in _edge_indices(lines)
                if len(lines[index].strip()) <= RUNNING_HEAD_MAX_LEN
            }
        )

    threshold = max(RUNNING_HEAD_MIN_PAGES, ceil(len(pages) * RUNNING_HEAD_RATIO))
    furniture = {line for line, count in seen.items() if count >= threshold}
    if not furniture:
        return list(pages)
    return [pages[0]] + [
        _drop_edge_lines(page, lambda line, _: line in furniture) for page in pages[1:]
    ]


def drop_page_numbers(pages: Sequence[PageText]) -> list[PageText]:
    """Document-wide. Remove the printed page number where it stands alone.

    The printed number rarely equals the page's ordinal — front matter and cover
    pages shift it — so the offset between the two is measured first, and a line
    is only removed where it continues that sequence. A bare number that does not
    keep step with its neighbours is data, and stays.
    """
    offsets: Counter[int] = Counter()
    for page in pages:
        lines = page.text.split("\n")
        for index in _edge_indices(lines):
            printed = _page_number(lines[index])
            if printed is not None:
                offsets[printed - page.number] += 1

    if not offsets:
        return list(pages)
    offset, count = offsets.most_common(1)[0]
    if count < max(RUNNING_HEAD_MIN_PAGES, ceil(len(pages) * RUNNING_HEAD_RATIO)):
        return list(pages)

    return [
        _drop_edge_lines(page, lambda line, n=page.number: _page_number(line) == n + offset)
        for page in pages
    ]


def join_hyphenated_words(pages: Sequence[PageText]) -> list[PageText]:
    """Page-local. Put back together a word the right margin split in two.

    The hyphen goes with it. A compound that happened to break at its own hyphen
    loses that hyphen as a result — telling the two apart needs a dictionary, and
    losing a hyphen costs less than leaving half a word in the index.
    """
    return _map_pages(pages, lambda text: _HYPHEN_BREAK_RE.sub("", text))


def join_wrapped_lines(pages: Sequence[PageText]) -> list[PageText]:
    """Page-local. Rejoin lines the right margin wrapped, leaving the rest.

    Line length is the whole signal: a line that runs the full measure of its
    page was ended by the margin and continues below, while a short one ended
    where its author meant it to. That keeps headings, list items and table cells
    on their own lines — which matters downstream, since `domain.chunking` reads
    section headings off the start of a line.
    """
    return _map_pages(pages, _join_wrapped)


def collapse_whitespace(pages: Sequence[PageText]) -> list[PageText]:
    """Page-local. Reduce the spacing left behind to one blank line at most."""
    return _map_pages(pages, _collapse)


def straighten_quotes(pages: Sequence[PageText]) -> list[PageText]:
    """Page-local. Fold typographic quotes and apostrophes to their ASCII forms.

    Not in `DEFAULT_STEPS`: unlike everything there, this changes a character a
    reader would see. Add it when a corpus is searched by people typing straight
    quotes and the difference costs more than the fidelity is worth.
    """
    return _map_pages(pages, lambda text: text.translate(_QUOTE_MAP))


DEFAULT_STEPS: tuple[Step, ...] = (
    normalize_characters,
    # Both furniture steps run on the page as it was set, before any line is
    # joined to another and stops being a line of its own.
    drop_running_heads,
    drop_page_numbers,
    join_hyphenated_words,
    join_wrapped_lines,
    collapse_whitespace,
)


# ── Internals ─────────────────────────────────────────────────────────────────

def _edge_indices(lines: Sequence[str]) -> list[int]:
    """Where the first and last `EDGE_LINES` non-blank lines are, in order."""
    filled = [index for index, line in enumerate(lines) if line.strip()]
    band = dict.fromkeys(filled[:EDGE_LINES] + filled[-EDGE_LINES:])
    return list(band)


def _drop_edge_lines(page: PageText, is_furniture: Callable[[str, int], bool]) -> PageText:
    """Remove the lines in the page's edge band that `is_furniture` accepts."""
    lines = page.text.split("\n")
    edges = set(_edge_indices(lines))
    kept = [
        line
        for index, line in enumerate(lines)
        if not (index in edges and is_furniture(line.strip(), page.number))
    ]
    return replace(page, text="\n".join(kept))


def _page_number(line: str) -> int | None:
    match = _PAGE_NUMBER_RE.match(line.strip())
    return int(match.group(1)) if match else None


def _measure(lines: Sequence[str]) -> float:
    """The width of a page, as the length its typical line runs to.

    The median rather than the maximum or a high percentile, because a page
    often carries more than one measure — body text at one width, a footnote
    block set narrower and denser at another. Reading the widest of them as the
    page's measure puts the threshold above the body text and leaves the part of
    the page that most needs joining untouched. The median finds the measure most
    of the page is set to, and `WRAP_WIDTH_RATIO` leaves room for the rest.
    """
    lengths = sorted(len(line) for line in lines if line.strip())
    if not lengths:
        return 0.0
    return float(lengths[min(int(len(lengths) * WIDTH_PERCENTILE), len(lengths) - 1)])


def _join_wrapped(text: str) -> str:
    lines = text.split("\n")
    width = _measure(lines) * WRAP_WIDTH_RATIO
    if width < WRAP_MIN_WIDTH:
        # The page has no full-measure text on it — a table, a figure, a caption.
        return text

    joined: list[str] = []
    # Measured against the line as it was set, not against what it has been
    # joined into: a paragraph's short last line ends the run even though the
    # full-measure line above it started one.
    wrapped = False
    for line in lines:
        stripped = line.rstrip()
        if wrapped and joined and stripped.strip() and not _is_heading(stripped):
            joined[-1] = f"{joined[-1]} {stripped.lstrip()}"
        else:
            joined.append(stripped)
        wrapped = len(stripped) >= width
    return "\n".join(joined)


def _is_heading(line: str) -> bool:
    """Whether a line is set as a heading, and so never continues the one above.

    Capitals are the signal that survives extraction — a PDF's text layer keeps
    no indent, no leading and no type size. It matters beyond the reading: a
    heading absorbed into the paragraph above it stops starting a line, and
    `domain.chunking` reads sections off the starts of lines.
    """
    stripped = line.strip()
    return (
        bool(stripped)
        and len(stripped) <= RUNNING_HEAD_MAX_LEN
        and any(char.isalpha() for char in stripped)
        and stripped == stripped.upper()
    )


def _collapse(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
