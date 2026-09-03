"""Implementations of `ports.outbound.ModelRuntime` — one module per backend.

Named to match `llm_providers`, because the two are halves of one backend: the
provider asks a model for something, the runtime is how the setup screen sees
what is installed and moves it in and out of memory. `llm_providers.registry`
builds them as a pair, and a backend that cannot load models on request says so
here rather than pretending — see `model_runtimes.vmlx`.
"""
