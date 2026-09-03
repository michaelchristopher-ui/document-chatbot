"""Implementations of `ports.outbound.ConversationalAgent`.

`model_input` sits here rather than in `domain` because what it does — deciding
which messages a model is shown — is framework-shaped: it exists because
LangGraph owns the loop, and it goes wherever the agent driving that loop goes.
"""
