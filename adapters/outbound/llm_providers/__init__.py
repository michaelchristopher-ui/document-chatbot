"""Implementations of `ports.outbound.LLMProvider` — one module per backend.

Each one is the only module that knows its backend exists: the address, the
credential and the departures from the OpenAI schema are spelled there and
nowhere else. `registry` picks between them, and pairs each with the
`model_runtimes` module of the same name.
"""
