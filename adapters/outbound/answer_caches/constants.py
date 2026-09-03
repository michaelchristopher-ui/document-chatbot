"""Constants for `adapters.outbound.answer_caches`."""

from __future__ import annotations

# The envelope version `codec` writes and refuses to read anything but. Bump it
# when the stored shape changes: every entry written under the old one then
# fails to decode, the adapter reports a miss, and the question is answered
# live. A cache going cold after a deploy is the correct outcome of changing
# what it stores.
VERSION = 1

# The tag each replayable event is stored under.
_TEXT_DELTA = "text_delta"
_SOURCES_FOUND = "sources_found"
_CONFIDENCE = "confidence"
