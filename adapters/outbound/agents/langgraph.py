"""The ReAct loop, kept behind the `ConversationalAgent` port.

The retriever it searches with is application logic injected from outside, so
swapping LangGraph out means replacing this file alone.

The one place in the app that does not reach its backend through `LLMProvider`'s
plain-typed methods, because it cannot: LangGraph owns the loop here rather than
this file owning it, and it drives a model object it insists on holding itself.
So this asks the provider for that object instead — see `ChatModelProvider`.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterator, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from adapters.outbound.agents.model_input import drop_stale_retrieval, prepare_input
from domain.citations import CitationLedger, TitleLookup, format_citations
from domain.constants import (
    PROMPT_KEY_SEARCH_TOOL,
    PROMPT_KEY_SYSTEM,
    SEARCH_TOOL_DESCRIPTION,
    SYSTEM_PROMPT,
)
from domain.models import AnswerEvent, Citation, SourcesFound, TextDelta, TokensUsed
from ports.outbound import PromptLibrary, Retriever


class ChatModelProvider(Protocol):
    """A provider that will hand over its model object, not just answers from it.

    `create_react_agent` requires a `BaseChatModel`: it calls `.bind_tools()` to
    attach the search tool, composes the result into a runnable with the system
    prompt, and invokes that inside its own graph. A method returning text
    cannot stand in for any of it.

    Declared here rather than on `ports.outbound.LLMProvider` so the LangChain
    type stays in the one module that already depends on LangChain. Every
    provider that can back this app satisfies both — see `LMStudioProvider`.
    """

    def chat_model(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stream_usage: bool = False,
    ) -> BaseChatModel: ...


class LangGraphAgent:
    """Implements `ConversationalAgent`."""

    def __init__(
        self,
        retriever: Retriever,
        provider: ChatModelProvider,
        model: str,
        title_of: TitleLookup,
        prompts: PromptLibrary,
    ):
        self._retriever = retriever
        self._title_of = title_of
        self._prompts = prompts
        # One ledger per thread, so an index means the same passage for as long
        # as the model can see the search that produced it. Numbering that
        # restarted each question would leave the transcript holding two
        # different `[1]`s, and no way to tell which one an answer meant.
        self._ledgers: dict[str, CitationLedger] = {}
        self._ledger = CitationLedger(title_of)
        # Citations are handed back by exact tool output, so a ToolMessage in the
        # stream resolves to the ones that produced it without re-parsing text.
        self._citations_by_output: dict[str, tuple[Citation, ...]] = {}
        # Usage is asked for here and nowhere else: this is the only caller that
        # streams, and `TokensUsed` is what the interaction log records a turn's
        # cost from. What that flag has to do to the request, and why it is not
        # on by default, is the provider's business.
        #
        # Built once and reused across rebuilds below: it is a connection to the
        # backend, not part of what the prompt decides.
        self._model = provider.chat_model(model, temperature=0.0, stream_usage=True)
        # Outlives every rebuild, which is the point of holding it here: threads
        # are checkpointed against this instance, so a prompt republished
        # mid-conversation must not be the thing that forgets the conversation.
        self._checkpointer = InMemorySaver()
        # `Any` because the graph `create_react_agent` returns is a LangGraph
        # type this file has no reason to name, and `object` would be a lie —
        # `stream` is called on it below.
        self._agent: Any = None
        # The prompt pair the agent standing above was built from. Compared
        # rather than assumed unchanged, because both halves can be republished
        # while this process is answering.
        self._built_from: tuple[str, str] | None = None

    def _current_agent(self) -> Any:
        """The graph, rebuilt if either prompt it was compiled with has changed.

        `create_react_agent` takes the system prompt and the tool description as
        values and composes them into a runnable, so neither can be swapped on
        a graph that already exists — which is why this exists at all rather
        than the constructor simply reading them.

        Rebuilding is cheap and, more to the point, rare: the prompts come from
        a `PromptLibrary` that caches, so the pair is identical on almost every
        call and this is a tuple comparison. Nothing is lost when it does
        rebuild — the checkpointer, the model and the citation ledgers all live
        on `self` — so a conversation continues across it with its history and
        its passage numbering intact.
        """
        pair = (
            self._prompts.text(PROMPT_KEY_SYSTEM, SYSTEM_PROMPT),
            self._prompts.text(PROMPT_KEY_SEARCH_TOOL, SEARCH_TOOL_DESCRIPTION),
        )
        if self._agent is None or self._built_from != pair:
            system, description = pair
            self._agent = create_react_agent(
                model=self._model,
                tools=[self._build_tool(description)],
                checkpointer=self._checkpointer,
                prompt=system,
                # The checkpointer keeps the thread whole; this decides how much
                # of it the model is shown. Without it the blocks of every
                # earlier search stay in context and the model answers from them
                # instead of searching again — see `drop_stale_retrieval`.
                # Further steps go in this call, in the order the model's input
                # passes through them.
                pre_model_hook=prepare_input(drop_stale_retrieval),
            )
            self._built_from = pair

        return self._agent

    def _build_tool(self, description: str) -> StructuredTool:
        return StructuredTool.from_function(
            func=self._search,
            name="search_documents",
            description=description,
        )

    def _search(self, query: str) -> str:
        """One search, as the model sees it — and as the opening search runs it.

        A method rather than the closure it was, because it now has two callers
        and they must produce the same thing: `_citations_by_output` resolves a
        `ToolMessage` back to its citations by exact output, so a second way of
        formatting the same passages would leave that lookup empty.
        """
        citations = self._ledger.record(self._retriever.retrieve(query))
        output = format_citations(citations)
        self._citations_by_output[output] = citations
        return output

    def _opening_search(
        self, question: str
    ) -> "tuple[list[AnyMessage], tuple[Citation, ...]]":
        """Search for `question`, as a tool exchange the model starts the turn holding.

        Rule 1 of the system prompt asks the model to search before answering
        anything factual, and a local model does not reliably do it — it answers
        from whatever is already in front of it, which on a fresh turn is
        nothing and after `drop_stale_retrieval` is nothing again. Forcing the
        call at the provider is not open to us either: LM Studio accepts
        `tool_choice: "required"` and ignores it, and rejects naming a tool
        outright. So the turn opens with the search already run.

        Written into the thread rather than into the model's input alone: these
        are the passages the answer will cite, so they belong in the transcript
        the checkpointer keeps — and being an ordinary tool exchange is what
        makes `drop_stale_retrieval` clear them again when the next question
        arrives.

        The tool stays bound and the loop stays a ReAct loop: a multi-part
        question still searches again, which is what rule 9 asks for. Only the
        first search stops being the model's decision.

        Degrades to nothing if the search fails. The model is left where it was
        before this existed — able to call the tool itself, against a backend
        that will most likely fail that too — and a turn that could still answer
        is not lost to an exception raised before the model was even asked.
        """
        try:
            output = self._search(question)
        except Exception:
            return [], ()

        call_id = f"opening-{uuid.uuid4()}"
        return [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_documents",
                    "args": {"query": question},
                    "id": call_id,
                    "type": "tool_call",
                }],
            ),
            ToolMessage(
                content=output, tool_call_id=call_id, name="search_documents"
            ),
        ], self._citations_by_output.get(output, ())

    def stream(self, thread_id: str, question: str) -> Iterator[AnswerEvent]:
        # Numbering picks up where this thread left off; a new thread — which is
        # what clearing the chat makes — starts again at [1], exactly as the
        # checkpointed history it is answering from does.
        self._ledger = self._ledgers.setdefault(
            thread_id, CitationLedger(self._title_of)
        )
        self._citations_by_output.clear()
        config = {"configurable": {"thread_id": thread_id}}

        # Yielded here rather than left to the loop below: the graph streams the
        # messages its own nodes write, and this exchange is handed in as input.
        opening, cited = self._opening_search(question)
        if cited:
            yield SourcesFound(cited)

        # Resolved here, once per turn, rather than per chunk: a prompt that
        # changed mid-answer would otherwise rebuild the graph underneath the
        # stream it is being read from.
        agent = self._current_agent()

        for chunk, _metadata in agent.stream(
            {"messages": [{"role": "user", "content": question}, *opening]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk):
                # Usage arrives on a final chunk carrying no choices, so it comes
                # with empty content — once per model call, and a turn that
                # searches before answering makes several. `AIMessageChunk` and
                # not `AIMessage`: the run also ends with an aggregated message
                # holding the summed usage, and matching the subclass alone is
                # what keeps this from counting a turn twice.
                if chunk.usage_metadata:
                    yield TokensUsed(
                        prompt=chunk.usage_metadata.get("input_tokens", 0),
                        completion=chunk.usage_metadata.get("output_tokens", 0),
                    )
                if isinstance(chunk.content, str) and chunk.content:
                    yield TextDelta(chunk.content)
            elif isinstance(chunk, ToolMessage):
                citations = self._citations_by_output.get(chunk.content, ())
                if citations:
                    yield SourcesFound(citations)
