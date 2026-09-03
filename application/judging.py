"""Scoring answers already given, rather than answers on their way out.

Judging runs over the log on request, not inside the chat. A second model call in
the answer path would double what a reader waits for — minutes, on a machine
running a large local model — to produce a number nobody is waiting to read. The
turn is already written down; scoring it later costs the reader nothing.

The consequence worth knowing: a turn is unjudged until someone asks, so the
statistics show scores for the turns that have been scored and say so plainly
rather than implying the rest passed.
"""

from __future__ import annotations

from typing import Iterator

from domain.interactions import TurnRecord
from ports.outbound import AnswerJudge, InteractionLog


class JudgingService:
    """Scores unjudged turns, one at a time, reporting as it goes."""

    def __init__(self, log: InteractionLog, judge: AnswerJudge):
        self._log = log
        self._judge = judge

    def pending(self, limit: int) -> list[TurnRecord]:
        return self._log.unjudged(limit)

    def score(self, limit: int) -> Iterator[tuple[TurnRecord, str | None]]:
        """Judge up to `limit` turns, yielding each with its error, or None.

        A generator so a caller can draw progress: judging is slow enough that a
        batch of ten is a minute or more, and a page that showed nothing until the
        end would look hung.

        One turn's failure is yielded, not raised. A model that cannot parse one
        answer should not stop the other nine from being scored.
        """
        for turn in self._log.unjudged(limit):
            if turn.id is None:
                continue
            try:
                judgement = self._judge.assess(
                    turn.question, turn.answer, _sources(turn)
                )
            except Exception as exc:
                yield turn, f"{type(exc).__name__}: {exc}"
                continue
            self._log.record_judgement(turn.id, judgement)
            # Re-read rather than patch the local copy, so what the caller shows is
            # what was stored.
            yield turn, None


def _sources(turn: TurnRecord) -> list[str]:
    """The passages the answer cited, in the order it numbered them.

    Only cited passages carry their text — see `interaction_logs.sqlite` — so this is
    the whole of what the judge can read, and it is the right set: the question is
    whether the answer follows from what it claimed to be following.
    """
    cited = [r for r in turn.retrievals if r.cited and r.text]
    cited.sort(key=lambda r: r.citation_index or 0)
    seen: dict[int | None, str] = {}
    for retrieval in cited:
        seen.setdefault(retrieval.citation_index, retrieval.text or "")
    return list(seen.values())
