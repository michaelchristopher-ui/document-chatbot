"""What a cache can say about a question, how a new answer gets recorded, and
what it has been doing.

Kept in the domain rather than beside the adapter because both the port and the
service that consumes it are above the port line, and neither may name the
library that does the matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

from domain.models import AnswerEvent


@dataclass(frozen=True)
class AnswerFound:
    """An answer to a near-enough question, and how near it was.

    `events` is the whole answer as it was streamed the first time, in order, so
    replaying it puts the reader in the same position as the reader who paid for
    it. `similarity` is the cosine the match was made at — recorded against the
    turn, because a hit at 0.93 and a hit at 0.999 are different claims about
    whether the same question was really asked twice.
    """

    events: tuple[AnswerEvent, ...]
    similarity: float


@dataclass(frozen=True)
class AnswerAbsent:
    """Nothing to serve, and what to do with the answer that gets written.

    The callable is the unusual part, and it is here for a reason. Deciding
    nothing is cached costs a normalisation and an embedding of the question;
    storing the answer afterwards needs both again. Handing back a closure lets
    the adapter keep that work without the core learning the name of the thing
    that did it — the alternative is either a second embedding per miss or a
    library type crossing the port.

    Call it once, with the finished answer. A turn abandoned mid-stream must
    simply not call it: a half-written answer is not one to serve to the next
    person who asks.
    """

    remember: Callable[[Sequence[AnswerEvent]], None]

    # Best similarity among the entries that were there but not close enough, or
    # None when there was nothing to score against. The only honest way to pick
    # a threshold is the distribution of this across real misses, which is why
    # it is carried up to be recorded rather than logged and dropped.
    nearest: Optional[float] = None


AnswerLookup = Union[AnswerFound, AnswerAbsent]


@dataclass(frozen=True)
class CacheStatistics:
    """What the answer cache has been doing, and under what settings.

    Two groups of numbers with different lifetimes, and the page that draws them
    has to say which is which or it misleads.

    `lookups` through `errors` are **counted in this process since it started**.
    They are lost on restart, they are not shared between replicas, and they are
    not synchronised — Streamlit runs each session on its own thread, so two
    turns answering at once can lose an increment. Good enough to read; wrong to
    bill from.

    `entries` is **the live state of the store**, which for Redis means every
    answer held for this index right now, including ones written by a process
    that has since exited.

    `threshold` and `ttl_seconds` are here because none of the above means
    anything without them: a hit rate is a statement about a threshold, and
    moving it makes every earlier number incomparable.
    """

    lookups: int
    hits: int
    exact_hits: int
    misses: int
    stores: int
    errors: int

    entries: int
    threshold: float
    ttl_seconds: float

    # None when the store could not be reached to ask. Distinct from 0, which
    # says the cache is genuinely empty.
    entries_known: bool = True

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def semantic_hits(self) -> int:
        """Hits that needed the embedding model to find them.

        The number that says whether this is worth its keep. `exact_hits` are
        the ones a plain string comparison would also have caught, so the
        difference is what the similarity search is actually earning.
        """
        return max(0, self.hits - self.exact_hits)

    @property
    def semantic_hit_rate(self) -> float:
        return self.semantic_hits / self.lookups if self.lookups else 0.0

    @property
    def exact_hit_rate(self) -> float:
        return self.exact_hits / self.lookups if self.lookups else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.lookups if self.lookups else 0.0
