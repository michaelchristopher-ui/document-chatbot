"""What a document is called.

The filename is how a document is keyed — it is what chunks carry, what filters
scope by, and what the store dedupes on — but it is rarely what the document is
*called*. Archive exports bake the authors, journal, DOI and a content hash into
it, so a citation that shows the filename shows two hundred characters of
provenance instead of a name. A document usually declares a better one about
itself; this module picks between that and what the filename can be reduced to.
"""

from __future__ import annotations

import os

from domain.constants import (
    TITLE_MAX_LENGTH,
    TITLE_MIN_LENGTH,
    _DOCUMENT_EXTENSIONS,
    _LETTER_RE,
    _PLACEHOLDERS,
    _SEPARATOR_RE,
    _WHITESPACE_RE,
)


def normalize(raw: str) -> str:
    """Collapse the line breaks a title spans in the document it came from."""
    return _WHITESPACE_RE.sub(" ", raw or "").strip()


def is_usable(title: str) -> bool:
    """Whether `title` names the document, rather than merely occupying the field.

    Applied to anything a document claims about itself, since a claim is only
    worth preferring over the filename when it reads like a name.
    """
    if not TITLE_MIN_LENGTH <= len(title) <= TITLE_MAX_LENGTH:
        return False
    if not _LETTER_RE.search(title):
        return False
    lowered = title.lower()
    if lowered.endswith(_DOCUMENT_EXTENSIONS):
        return False
    return not any(lowered.startswith(placeholder) for placeholder in _PLACEHOLDERS)


def from_filename(filename: str) -> str:
    """The readable part of a filename: everything before the provenance run."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    candidate = normalize(_SEPARATOR_RE.split(stem)[0]) or normalize(stem)
    if " " not in candidate:
        # A slug — `quarterly_report-2024` — reads as a title once punctuation
        # that stood in for spaces is spaces again.
        candidate = normalize(candidate.replace("_", " ").replace("-", " "))
    return candidate or filename


def resolve(declared: str, filename: str) -> str:
    """The title to show for a document, given what it declares about itself."""
    declared = normalize(declared)
    return declared if is_usable(declared) else from_filename(filename)
