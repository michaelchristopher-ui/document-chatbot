"""Driven adapters — implementations of the ports in `ports.outbound`.

One package per port, named for what it implements rather than for what
implements it, so the question "what can back this?" is answered by a directory
listing. A port with one implementation still gets a package: the next one lands
beside it rather than reopening the question of where it goes.

Two of them hold a `registry` as well — `llm_providers` and `vector_stores`, the
ports with a choice to make — and that module is the only place a backend is
named. Everything above these packages holds a port and knows none of this.
"""
