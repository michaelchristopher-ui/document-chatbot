"""Recording a turn, without the code that answers it knowing it is recorded.

Two decorators and the channel between them, kept in one module so the place a
search is collected and the place it is read are twenty lines apart.

The channel is the interesting part. A search happens several layers below the
call that will write the row — inside the agent's ReAct loop, in a tool the model
decided to invoke — and LangGraph runs that tool on a worker thread as soon as
the model asks for two searches at once. So the turn hands a list *down* through
a `ContextVar` instead of trying to collect results on the way back up.

Why that specific mechanism, since three plausible ones do not work here:

- An attribute on the decorator is wrong. `@st.cache_resource` makes one
  `Application` per model configuration, shared by every browser session, so two
  turns can be inside the same object at the same moment and each would see the
  other's searches.
- A `threading.local` is wrong. The tool worker thread is not the thread that set
  it, so a multi-search turn — exactly the case worth recording — would find it
  empty and log nothing.
- A `ContextVar` holding an immutable value is wrong. LangGraph gives the worker
  thread a *copy* of the context, so a `set()` inside it would not come back.

A `ContextVar` holding a mutable list is right: the copy still points at the same
list, so an `append` from the worker lands in the turn that started it.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Iterator, List, Optional

from domain.filters import MetadataFilter
from domain.interactions import (
    SearchOutcome,
    TokenUsage,
    TurnContext,
    build_turn,
)
from domain.models import AnswerEvent, Citation, Passage, SourcesFound, TextDelta, TokensUsed
from ports.inbound import AnswerQuestions
from ports.outbound import InteractionLog, Retriever

# Where a turn in flight collects the searches made on its behalf, or None when
# nothing is being recorded. Module level rather than an attribute, because the
# object that would hold the attribute is shared by every session at once.
_TURN_SEARCHES: "ContextVar[Optional[List[SearchOutcome]]]" = ContextVar(
    "turn_searches", default=None
)


class RecordingRetriever:
    """Implements `Retriever` by delegating, noting each search on the way past.

    Silent when no turn is collecting — a search run from a test or a script is
    still a search, but there is no row for it to belong to.
    """

    def __init__(self, retriever: Retriever):
        self._retriever = retriever

    def retrieve(self, query: str, where: MetadataFilter | None = None) -> list[Passage]:
        passages = self._retriever.retrieve(query, where)
        collecting = _TURN_SEARCHES.get()
        if collecting is not None:
            collecting.append(SearchOutcome(query=query, passages=tuple(passages)))
        return passages


class RecordedChat:
    """Implements `AnswerQuestions` by delegating, and writes down what it saw.

    A decorator rather than a change to `ChatService`, because recording is not
    part of answering and an answer has to survive a log that fails.

    It has to wrap `ChatService` rather than the other way round: that class
    merges every search's citations into each `SourcesFound` it emits, and the
    numbering in those merged citations is what tells a retrieved passage apart
    from one the answer went on to cite.
    """

    def __init__(self, chat: AnswerQuestions, log: InteractionLog, context: TurnContext):
        self._chat = chat
        self._log = log
        self._context = context

    def ask(self, thread_id: str, question: str) -> Iterator[AnswerEvent]:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        started = time.perf_counter()
        first_token: float | None = None
        parts: list[str] = []
        citations: tuple[Citation, ...] = ()
        usage = TokenUsage()
        outcomes: list[SearchOutcome] = []
        error: str | None = None

        # Set inside a generator, which shares its caller's context — and the
        # caller is a fresh Streamlit script thread whose context starts empty.
        # So each turn is isolated from every other without any locking.
        token = _TURN_SEARCHES.set(outcomes)
        try:
            for event in self._chat.ask(thread_id, question):
                if isinstance(event, TextDelta):
                    if first_token is None:
                        first_token = time.perf_counter()
                    parts.append(event.text)
                elif isinstance(event, SourcesFound):
                    # Already cumulative, so the last one seen is the whole turn.
                    citations = event.citations
                elif isinstance(event, TokensUsed):
                    usage = usage + TokenUsage(event.prompt, event.completion)
                yield event
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._release(token)
            # In `finally` so it also runs on `GeneratorExit` — a rerun that
            # abandons a half-streamed answer still leaves the partial answer and
            # the searches it managed, which is a turn worth having.
            self._write(
                thread_id=thread_id,
                created_at=created_at,
                question=question,
                answer="".join(parts),
                latency_ms=_elapsed_ms(started, time.perf_counter()),
                first_token_ms=(
                    _elapsed_ms(started, first_token) if first_token else None
                ),
                usage=usage,
                outcomes=outcomes,
                citations=citations,
                error=error,
            )

    @staticmethod
    def _release(token) -> None:
        try:
            _TURN_SEARCHES.reset(token)
        except ValueError:
            # The token belongs to another context, which happens when the
            # generator is finalised by the collector on a different thread.
            # Clearing is as good as resetting: nothing else shares this context.
            _TURN_SEARCHES.set(None)

    def _write(self, **fields) -> None:
        """Record the turn, or give up quietly.

        Nothing here is worth failing an answer the reader has already been given,
        and raising from the `finally` above would replace a real exception with a
        bookkeeping one.
        """
        try:
            self._log.record(build_turn(context=self._context, **fields))
        except Exception:
            pass


def _elapsed_ms(started: float, ended: float) -> int:
    return max(0, int((ended - started) * 1000))
