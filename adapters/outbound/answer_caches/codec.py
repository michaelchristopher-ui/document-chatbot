"""Answer events to JSON and back, so a stored answer outlives the process.

In memory an `AnswerEvent` is a frozen dataclass and costs nothing to keep; in
Redis it has to become bytes, and the obvious way to do that is the wrong one.
`pickle` would work today and then break twice: it executes whatever it reads,
and it is a promise that these dataclasses never change shape again — a renamed
field on `Passage` would make every entry written before the rename either an
error or, worse, a half-built object.

So: an explicit envelope with a version on it. When the shape changes, bump
`VERSION`; every entry written under the old one then fails to decode, the
adapter reports a miss, and the question is answered live. A cache going cold
after a deploy is the correct outcome of changing what it stores.

Only the three events a replay emits are handled. `TokensUsed` is deliberately
not among them — see `application.caching`, which drops it before storing,
because nothing is spent the second time a question is asked and a replayed
usage event would bill the reader for someone else's tokens.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from adapters.outbound.answer_caches.constants import (
    VERSION,
    _CONFIDENCE,
    _SOURCES_FOUND,
    _TEXT_DELTA,
)
from domain.models import (
    AnswerConfidence,
    AnswerEvent,
    ChunkMetadata,
    Citation,
    Passage,
    RetrievalOrigin,
    SourcesFound,
    TextDelta,
)


class UnreadableAnswer(Exception):
    """An envelope this version cannot read, or an event it cannot store.

    Raised rather than repaired, and caught by the store beneath it as a cache
    miss. Guessing at half an answer would serve the reader a citation list
    assembled from whatever survived the version change.
    """


def encode(events: Sequence[AnswerEvent]) -> str:
    """One answer as a JSON string.

    `asdict` does the nesting — `Citation` holds a `Passage` holds a
    `ChunkMetadata` — so this only has to name the tag each event is read back
    under.
    """
    payload = [{"type": _tag(event), "data": asdict(event)} for event in events]
    return json.dumps({"v": VERSION, "events": payload})


def decode(raw: str) -> tuple[AnswerEvent, ...]:
    """The inverse of `encode`, strict about the version and the shape."""
    envelope = json.loads(raw)
    if not isinstance(envelope, Mapping):
        raise UnreadableAnswer("envelope is not an object")
    if envelope.get("v") != VERSION:
        raise UnreadableAnswer(
            f"envelope version {envelope.get('v')!r}, this build reads {VERSION}"
        )
    events = envelope.get("events")
    if not isinstance(events, list):
        raise UnreadableAnswer("envelope carries no event list")
    return tuple(_event(item) for item in events)


def _tag(event: AnswerEvent) -> str:
    if isinstance(event, TextDelta):
        return _TEXT_DELTA
    if isinstance(event, SourcesFound):
        return _SOURCES_FOUND
    if isinstance(event, AnswerConfidence):
        return _CONFIDENCE
    raise UnreadableAnswer(f"{type(event).__name__} is not a storable answer event")


def _event(item: Any) -> AnswerEvent:
    if not isinstance(item, Mapping):
        raise UnreadableAnswer("event is not an object")
    tag = item.get("type")
    data = item.get("data")
    if not isinstance(data, Mapping):
        raise UnreadableAnswer(f"event {tag!r} carries no data")

    try:
        if tag == _TEXT_DELTA:
            return TextDelta(text=str(data["text"]))
        if tag == _SOURCES_FOUND:
            return SourcesFound(
                citations=tuple(_citation(c) for c in data["citations"])
            )
        if tag == _CONFIDENCE:
            return AnswerConfidence(**dict(data))
    except (KeyError, TypeError, ValueError) as exc:
        # A field this build does not know, or one it expects and did not get.
        # Both mean the envelope was written by other code.
        raise UnreadableAnswer(f"event {tag!r} does not fit this build: {exc}") from exc
    raise UnreadableAnswer(f"unknown event type {tag!r}")


def _citation(data: Any) -> Citation:
    return Citation(
        index=int(data["index"]),
        passage=_passage(data["passage"]),
        title=str(data["title"]),
    )


def _passage(data: Any) -> Passage:
    origin = data.get("origin")
    metadata = data.get("metadata")
    return Passage(
        text=str(data["text"]),
        page=int(data["page"]),
        source_file=str(data["source_file"]),
        strategy=str(data.get("strategy", "")),
        metadata=ChunkMetadata(**dict(metadata)) if metadata else ChunkMetadata(),
        score=data.get("score"),
        origin=RetrievalOrigin(**dict(origin)) if origin else None,
    )
