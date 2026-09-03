from __future__ import annotations

from typing import Iterator

from domain.citations import merge_citations
from domain.confidence import assess
from domain.models import AnswerEvent, Citation, SourcesFound, TextDelta
from ports.outbound import ConversationalAgent


class ChatService:
    """Answers questions in a conversation thread and keeps citations complete."""

    def __init__(self, agent: ConversationalAgent):
        self._agent = agent

    def ask(self, thread_id: str, question: str) -> Iterator[AnswerEvent]:
        """Stream the answer, and close with how far it stands up.

        The confidence event comes last because it cannot come earlier: two of
        its three readings are about the finished text — which of its claims
        carry markers, which parts of the question it reached — and neither is a
        fact about a half-written sentence. It costs no model call, so the reader
        waits no longer for it than for the last token.

        Here rather than in a decorator beside `RecordedChat`, because this is
        part of the answer rather than a note taken about one: it is returned to
        whoever asked, and it is built from the merged citations this class
        already keeps. A turn abandoned mid-stream simply never reaches it.
        """
        gathered: tuple[Citation, ...] = ()
        written: list[str] = []
        for event in self._agent.stream(thread_id, question):
            if isinstance(event, SourcesFound):
                # A multi-part question triggers several searches, and the answer
                # cites across all of them, so the reader needs every search's
                # results rather than the latest one's.
                gathered = merge_citations(gathered, event.citations)
                yield SourcesFound(gathered)
            else:
                if isinstance(event, TextDelta):
                    written.append(event.text)
                yield event
        yield assess(question, "".join(written), gathered)
