"""Serving a question someone has already asked, without asking the model again.

Two decorators, kept in one module because they are two halves of one policy:
`CachedChat` decides what may be served, and `InvalidatingIngest` decides when
none of it may be any more. Both are decorators for the same reason
`RecordedChat` next door is one — answering is one job, and deciding not to
answer is another. `ChatService` and `IngestionService` are untouched by this
file.

Why it is a generator rather than a call around one. The port streams, and the
reader is watching tokens arrive, so a cached answer has to arrive the same way
— which means collecting the events on the way past on a miss, and replaying
them on a hit. The library's own `cached` decorator cannot be used here: it
would store the generator object, a one-shot iterator that is already exhausted
by the time anyone hits it.

**The first-turn gate is the load-bearing part of this file.** The agent is
conversational — `LangGraphAgent` keeps each thread's history in a checkpointer
— and a cache hit does not reach the agent, so a served answer never enters the
thread it was served into. Two things follow, and both are wrong: the next
question in that thread sees a hole where this exchange should be, and a
question that only means something in context ("what about the second one?")
could be answered from an entry another thread wrote. So only a thread's *first*
question is cached. That is where the value is anyway: two people opening the
app and asking the same thing is the case worth catching, and a follow-up is
cheap to get wrong and expensive to be wrong about.
"""

from __future__ import annotations

import threading
from typing import Callable, Iterator, Sequence

from domain.answers import AnswerFound
from domain.models import (
    AnswerEvent,
    DocumentIndexing,
    IngestionEvent,
    TokensUsed,
)
from ports.inbound import AnswerQuestions, IngestDocuments
from ports.outbound import AnswerCache


class CachedChat:
    """Implements `AnswerQuestions` by replaying an answer given once already."""

    def __init__(self, chat: AnswerQuestions, cache: AnswerCache):
        self._chat = chat
        self._cache = cache
        # Which threads this process has already answered in. It mirrors the
        # agent's own checkpointer, which is process-local too, so the two agree
        # on what "first question" means without either consulting the other.
        # Bounded in practice by the number of browser sessions since restart.
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def ask(self, thread_id: str, question: str) -> Iterator[AnswerEvent]:
        if not self._first_question(thread_id):
            yield from self._chat.ask(thread_id, question)
            return

        found = self._cache.lookup(question)
        if isinstance(found, AnswerFound):
            yield from found.events
            return

        collected: list[AnswerEvent] = []
        for event in self._chat.ask(thread_id, question):
            # `TokensUsed` is dropped from what gets stored, not from what gets
            # yielded: this turn really did spend them and the log should say so,
            # while a replay of it spends nothing and must not claim otherwise.
            if not isinstance(event, TokensUsed):
                collected.append(event)
            yield event

        # Only reached when the stream finished and the caller consumed all of
        # it. A rerun that abandons a half-written answer raises `GeneratorExit`
        # at the yield above and never arrives here, which is the intent: a
        # partial answer is not one to serve to the next person who asks.
        self._remember(found.remember, collected)

    def _first_question(self, thread_id: str) -> bool:
        """Whether this thread has been answered in yet, marking it either way.

        Under a lock because Streamlit runs each session's script on its own
        thread, and two arriving at once must not both be told they are first.
        """
        with self._lock:
            if thread_id in self._seen:
                return False
            self._seen.add(thread_id)
            return True

    @staticmethod
    def _remember(
        remember: Callable[[Sequence[AnswerEvent]], None],
        events: list[AnswerEvent],
    ) -> None:
        """Store the answer, or give up quietly.

        The reader already has this answer; nothing about writing it down is
        worth taking that away from them. The same trade `RecordedChat._write`
        makes, and for the same reason.
        """
        try:
            remember(events)
        except Exception:
            pass


class InvalidatingIngest:
    """Implements `IngestDocuments`, emptying the answer cache when the corpus moves.

    The invalidation that actually matters, and the one a TTL cannot do. An
    answer is a reading of the documents as they stood when it was written: the
    question does not change, so nothing about the question can tell the cache
    that the answer has. A cached "the documents do not cover that" outlives the
    document that arrived to cover it, and goes on being served, confidently,
    until it expires.

    `DocumentIndexing` is the signal, because it is emitted only when chunks are
    genuinely on their way to the store. An ingest where every document was
    unchanged — the common case, and the whole point of the ledger — emits none
    of them and correctly leaves the cache alone. `IngestionReport` cannot answer
    this: it lists every document with a status whether or not this run touched
    it.

    Whole namespace rather than the entries that moved, because there is no
    relation to be had between a question and the documents that would have
    answered it. Which is a real cost — one new file empties every answer — and
    still the right trade: the alternative is serving an answer that was true
    before the ingest.
    """

    def __init__(self, ingestion: IngestDocuments, cache: AnswerCache):
        self._ingestion = ingestion
        self._cache = cache

    def ingest_all(self) -> Iterator[IngestionEvent]:
        indexed_something = False
        try:
            for event in self._ingestion.ingest_all():
                if isinstance(event, DocumentIndexing):
                    indexed_something = True
                yield event
        finally:
            # In `finally` so an ingest abandoned or failed partway still
            # invalidates: the documents it managed before stopping are in the
            # store, and answers written before them are already stale.
            if indexed_something:
                self._forget()

    def _forget(self) -> None:
        """Empty the cache, or give up quietly.

        An unreachable Redis must not fail an ingest that has already written
        every chunk it read. What it costs is answers served from a stale cache
        until they expire, which is what the TTL is the backstop for.
        """
        try:
            self._cache.forget_all()
        except Exception:
            pass
