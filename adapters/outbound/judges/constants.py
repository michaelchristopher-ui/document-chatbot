"""Constants for `adapters.outbound.judges`."""

from __future__ import annotations

import re

# Long enough for a score and a handful of short claims; the judge has no reason
# to write an essay, and a cap keeps one confused reply from stalling a batch.
MAX_TOKENS = 600
# A judge that answers differently each time cannot be compared across runs.
TEMPERATURE = 0.0

_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
