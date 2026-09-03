"""What the model is shown on each call, assembled by a chain of steps.

The list a thread holds and the list the model is given are not the same thing.
The checkpointer keeps every message the thread ever held — that is what makes a
follow-up question a follow-up — while the model is shown whatever answering
*this* question calls for. This module is the seam between the two, and
`prepare_input` is the node `create_react_agent` runs before every model call to
cross it.

A chain of responsibility rather than a list of transforms: each step is handed
the messages and the rest of the chain, so it decides what the steps behind it
see, when they see it, and whether they run at all. A step that summarised a long
thread could answer from its own cache and stop the chain there; one that trims
hands a shorter list on and lets the rest work over that. Adding a step is
writing a function of this shape and naming it in the `prepare_input` call — see
`LangGraphAgent`.

LangChain message types live here because the messages themselves do: this is a
step in LangGraph's loop, and it goes wherever `adapters.outbound.agents.langgraph`
goes.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

# The rest of the chain, as a step sees it: hand it messages, get back the input
# the model will be called with.
Continue = Callable[[Sequence[AnyMessage]], "list[AnyMessage]"]


class MessageStep(Protocol):
    """One link in the chain.

    Call `following(messages)` to pass a — possibly rewritten — list on, and
    return what comes back. Returning without calling it ends the chain there,
    which is the point of the pattern: a step that has already decided what the
    model should see does not have to trust the ones behind it to leave it alone.
    """

    def __call__(
        self, messages: Sequence[AnyMessage], following: Continue
    ) -> "list[AnyMessage]": ...


def prepare_input(*steps: MessageStep) -> Callable[[dict], dict]:
    """The `pre_model_hook` node that runs `steps`, in order, before each call.

    Returns `llm_input_messages` rather than `messages`: the first is what the
    agent node is given, the second is what the thread keeps. Writing the second
    would make every step destructive — the transcript would lose whatever was
    dropped, and the turn after it would have nothing left to work from.
    """
    handler = chain(*steps)

    def hook(state: dict) -> dict:
        return {"llm_input_messages": handler(state["messages"])}

    return hook


def chain(*steps: MessageStep) -> Continue:
    """Fold `steps` into one handler, the first step outermost.

    The tail is `list`: a chain that runs out of steps has arrived at the input,
    and copies it so no step is handed the thread's own list to mutate.
    """
    handler: Continue = list
    for step in reversed(steps):
        handler = _link(step, handler)
    return handler


def _link(step: MessageStep, following: Continue) -> Continue:
    """Bind one step to the rest of the chain.

    A function rather than a lambda written inline in the loop above, which would
    close over the loop variable and leave every link calling the last step.
    """
    return lambda messages: step(messages, following)


# ── Steps ─────────────────────────────────────────────────────────────────────

def drop_stale_retrieval(
    messages: Sequence[AnyMessage], following: Continue
) -> "list[AnyMessage]":
    """Keep what earlier turns asked and answered; drop what they retrieved.

    Without this, the numbered blocks of every search a thread has ever run stay
    in front of the model for the rest of that thread — and it stops searching.
    The system prompt tells it to answer from the blocks `search_documents`
    returned, and blocks returned three questions ago satisfy that reading of it.
    So the first document answered from becomes the whole corpus: a later
    question that document happens to cover is answered from its passages, and
    one it does not cover is refused outright, both without a search ever being
    run.

    Rule 6 of the prompt already tells the model the reader is no longer shown
    those blocks. This makes it true of the model too, leaving the tool as the
    only way to reach a passage. The transcript is not touched — questions and
    answers stay, so a follow-up still reads as one — and neither is the turn in
    flight, which is mid-loop over searches it is about to answer from.
    """
    current = _current_turn(messages)
    return following([*_without_retrieval(messages[:current]), *messages[current:]])


def _current_turn(messages: Sequence[AnyMessage]) -> int:
    """Where the turn in flight begins — the last question asked.

    Everything from there on belongs to answering it: the searches the model has
    already run this turn and their results, which it must keep seeing to finish.
    Everything before it is a turn already answered.

    A list holding no question at all is left alone rather than emptied: there is
    nothing to answer, so there is nothing this step can usefully say about it.
    """
    for index in reversed(range(len(messages))):
        if isinstance(messages[index], HumanMessage):
            return index
    return 0


def _without_retrieval(messages: Sequence[AnyMessage]) -> "list[AnyMessage]":
    """`messages` with every tool call and every tool result taken out.

    The two halves go together or not at all: a provider rejects a tool call with
    no result beside it, and `create_react_agent` validates for exactly that
    before it calls the model. An AI message that carried text alongside its tool
    call keeps the text, rebuilt as a plain message — the call it made is gone,
    and a copy carrying it would fail that validation on the way past.
    """
    kept: "list[AnyMessage]" = []
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            text = message.content if isinstance(message.content, str) else ""
            if text:
                kept.append(AIMessage(content=text))
            continue
        kept.append(message)
    return kept
