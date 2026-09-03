"""Metadata scoping for retrieval.

The two retrieval arms enforce a filter by different means — the vector store
compiles it to a backend expression, the in-memory keyword index evaluates
`matches` directly — so the filter itself stays a plain domain value that knows
neither.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from domain.models import Passage


@dataclass(frozen=True)
class MetadataFilter:
    """A conjunction of scopes; empty tuples mean "no constraint on this field".

    `pages` is an inclusive (first, last) range.
    """

    source_files: Tuple[str, ...] = ()
    sections: Tuple[str, ...] = ()
    strategies: Tuple[str, ...] = ()
    pages: Optional[Tuple[int, int]] = None

    @property
    def is_empty(self) -> bool:
        return not (self.source_files or self.sections or self.strategies or self.pages)

    def matches(self, passage: Passage) -> bool:
        if self.source_files and passage.source_file not in self.source_files:
            return False
        if self.sections and passage.metadata.section not in self.sections:
            return False
        if self.strategies and passage.strategy not in self.strategies:
            return False
        if self.pages and not self.pages[0] <= passage.page <= self.pages[1]:
            return False
        return True
