"""Driving ports — the use cases a driving adapter (UI, CLI, HTTP) may invoke."""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Protocol, Tuple

from domain.answers import CacheStatistics
from domain.interactions import TurnRecord
from domain.models import AnswerEvent, IngestionEvent, IngestionOutcome


class IngestDocuments(Protocol):
    def ingest_all(self) -> Iterator[IngestionEvent]:
        """Index every available document, reporting progress as it goes.

        A stream rather than a return value for the same reason `ask` is one:
        the work takes long enough that whoever asked for it needs to be told
        what it is doing — though it may also finish in an instant, since a
        document already indexed and unchanged since is left alone. Lazy with
        it — nothing is read until the events are drawn, so `NoDocumentsFound`
        surfaces from the first `next` rather than from the call — and the last
        event is always `IngestionFinished`, carrying the report for the whole
        run.
        """


class ListDocuments(Protocol):
    def documents(self) -> list[IngestionOutcome]:
        """Every document on record, including ones an earlier run indexed."""


class AnswerQuestions(Protocol):
    def ask(self, thread_id: str, question: str) -> Iterable[AnswerEvent]: ...


class ScoreAnswers(Protocol):
    """Judging answers already recorded. Absent when no judge model is configured."""

    def pending(self, limit: int) -> list[TurnRecord]:
        """Answered turns nothing has scored yet."""

    def score(self, limit: int) -> Iterator[Tuple[TurnRecord, Optional[str]]]:
        """Judge up to `limit` turns, yielding each with its error, or None."""


class ViewCacheStatistics(Protocol):
    """Reading what the answer cache has been doing.

    Its own port rather than a method on `ViewStatistics`, because the two
    answer to different stores with different lifetimes: one reads a durable
    log of turns, this one reads counters held in this process and a keyspace
    that expires itself. Absent altogether when no cache is configured, which
    is what the UI keys the page's existence off — a page reporting zeroes
    would read as a cache that is not working rather than one that is not
    there.
    """

    def statistics(self) -> CacheStatistics: ...


class ViewStatistics(Protocol):
    def turns(self, limit: int) -> list[TurnRecord]:
        """The most recent turns to read statistics from, oldest first.

        Raw turns rather than a summary: what is worth computing over them is
        `domain.statistics`' business, and a port that returned one fixed summary
        would have to grow a method for every question later asked of it.
        """
