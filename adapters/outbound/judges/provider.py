"""Implements `AnswerJudge` against whatever backend the provider reaches.

One completion per answer, no tools and no streaming: the judge reads text that
has already been written and returns a number. It is deliberately a separate
model from the one that answers — a small fast model judges well enough here,
and asking the author to mark its own work is worth less than asking anyone
else. It shares the provider with everything else, so that second model costs a
model name, not a second connection.

The reply is parsed leniently. Local models wrap JSON in prose or a code fence
often enough that insisting on a clean object would fail turns for a formatting
habit rather than for anything about the answer. What is *not* lenient is the
score itself: an unreadable reply raises, so the turn stays unjudged rather than
recording a number nobody produced.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Sequence

from adapters.outbound.judges.constants import MAX_TOKENS, TEMPERATURE, _OBJECT_RE
from domain.constants import (
    JUDGE_PROMPT,
    JUDGE_REQUEST,
    PROMPT_KEY_JUDGE,
    PROMPT_KEY_JUDGE_REQUEST,
)
from domain.interactions import Judgement
from domain.models import ChatMessage
from ports.outbound import LLMProvider, PromptLibrary


class JudgeUnreadable(Exception):
    """The judge replied with something that holds no score."""


def _extract(reply: str) -> dict:
    """The JSON object inside `reply`, however it was wrapped."""
    match = _OBJECT_RE.search(reply)
    if not match:
        raise JudgeUnreadable(f"no JSON object in reply: {reply[:200]!r}")
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise JudgeUnreadable(f"malformed JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise JudgeUnreadable(f"expected an object, got {type(parsed).__name__}")
    return parsed


def _faithfulness(parsed: dict) -> float:
    try:
        score = float(parsed["faithfulness"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JudgeUnreadable(f"no usable faithfulness in {parsed!r}") from exc
    # Models occasionally answer on a 0-100 scale despite the instruction. Reading
    # 85 as 0.85 recovers the intent; clamping keeps anything else in range.
    if score > 1.0:
        score = score / 100.0
    return min(1.0, max(0.0, score))


def _unsupported(parsed: dict) -> tuple[str, ...]:
    claims = parsed.get("unsupported") or ()
    if isinstance(claims, str):
        claims = [claims]
    return tuple(str(claim).strip() for claim in claims if str(claim).strip())


def _format_sources(sources: Sequence[str]) -> str:
    return "\n\n".join(f"[{n}] {text}" for n, text in enumerate(sources, start=1))


class ProviderJudge:
    """Implements `AnswerJudge`."""

    def __init__(self, provider: LLMProvider, model: str, prompts: PromptLibrary):
        self._provider = provider
        self._model = model
        self._prompts = prompts

    @property
    def model(self) -> str:
        return self._model

    def assess(self, question: str, answer: str, sources: Sequence[str]) -> Judgement:
        if not sources:
            # Nothing to check the answer against. That is a fact about the turn,
            # not a verdict on it, so it is not the judge's to invent.
            raise JudgeUnreadable("the turn cited no passages")

        # `str.format` and not the library's `{{placeholder}}` rendering: this
        # template's three fields are filled per judgement from the turn being
        # scored, which is the caller's job and not the registry's.
        system = self._prompts.text(PROMPT_KEY_JUDGE, JUDGE_PROMPT)
        request = self._prompts.text(PROMPT_KEY_JUDGE_REQUEST, JUDGE_REQUEST)
        reply = self._provider.chat(
            self._model,
            [
                ChatMessage("system", system),
                ChatMessage(
                    "user",
                    request.format(
                        question=question,
                        answer=answer,
                        sources=_format_sources(sources),
                    ),
                ),
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        parsed = _extract(reply)
        return Judgement(
            faithfulness=_faithfulness(parsed),
            unsupported=_unsupported(parsed),
            model=self._model,
            judged_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
