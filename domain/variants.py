"""What makes one index reusable by a later run, and separate from another.

A vector is only comparable to vectors from the same embedding model, and pages
cut by one chunking strategy are a different corpus from the same pages cut by
another. So the pair does not *qualify* an index — it *names* one. Each gets a
collection of its own and a ledger of its own, they sit side by side in the same
database, and going back to a combination used before is a lookup rather than a
re-ingest.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from domain.constants import READABLE_MAX


def _readable(value: str) -> str:
    """`value` reduced to the characters a collection name may be spelled with."""
    return re.sub(r"[^0-9a-z]+", "_", value.lower()).strip("_") or "unnamed"


@dataclass(frozen=True)
class IndexVariant:
    """The embedding model and chunking strategy one index was built with."""

    embed_model: str
    chunking_strategy: str

    @property
    def slug(self) -> str:
        """A name for this variant: readable, and unique to the exact pair.

        The digest is not decoration. Sanitising a model id throws characters
        away, so two ids that differ only in punctuation read the same
        afterwards — and an index answering for the wrong model is worse than
        one that has to be rebuilt.
        """
        exact = "\x00".join((self.embed_model, self.chunking_strategy)).encode()
        digest = hashlib.sha256(exact).hexdigest()[:8]
        readable = f"{_readable(self.embed_model)}_{_readable(self.chunking_strategy)}"
        return f"{readable[:READABLE_MAX]}_{digest}"

    def collection(self, base: str) -> str:
        """Where this variant's chunks live, under the configured base name."""
        return f"{base}_{self.slug}"
