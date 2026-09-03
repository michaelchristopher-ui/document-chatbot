"""Constants for `adapters.outbound.llm_providers`."""

from __future__ import annotations

import os

# Backend names, as `LLM_BACKEND` and `RERANK_BACKEND` spell them, and as
# `registry` and `constants.BACKENDS` are both keyed by.
LM_STUDIO = "lmstudio"
VMLX = "vmlx"
LLAMA_CPP = "llamacpp"

# What a probe embeds to learn a model's dimension — the shortest input that is
# still an input. Shared: every backend answers the same question the same way.
PROBE_TEXT = "test"


# ── LM Studio ─────────────────────────────────────────────────────────────────

LM_STUDIO_LABEL = "LM Studio"
# LM Studio authenticates nothing, but the OpenAI client refuses to start
# without a key, so it gets one that means nothing to either end.
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_RERANK_TIMEOUT = 60


# ── llama.cpp ─────────────────────────────────────────────────────────────────

LLAMA_CPP_LABEL = "llama.cpp"
# llama-server authenticates nothing unless it was started with `--api-key`, but
# the OpenAI client refuses to start without a key, so an unguarded server gets
# one it will not look at. A guarded one needs the real value in the environment.
# `or` rather than a default: an orchestrator that lists this variable sets it
# to the empty string when nobody filled it in, and an empty key is not a
# missing key to the OpenAI client — it refuses to build a client at all.
LLAMA_CPP_API_KEY = os.getenv("LLAMA_API_KEY") or "not-needed"
# No download hides inside the first call the way it can on vMLX: llama-server
# loads its model before it binds the port, so a request that reaches this is
# inference and nothing else.
LLAMA_CPP_RERANK_TIMEOUT = 60


# ── vMLX ──────────────────────────────────────────────────────────────────────

VMLX_LABEL = "vMLX"
# vMLX authenticates nothing unless it was started with `--api-key`, but the
# OpenAI client refuses to start without a key, so an unguarded server gets one
# it will not look at. A guarded one needs the real value in the environment.
# `or` rather than a default: an orchestrator that lists this variable sets it
# to the empty string when nobody filled it in, and an empty key is not a
# missing key to the OpenAI client — it refuses to build a client at all.
VMLX_API_KEY = os.getenv("VMLX_API_KEY") or "not-needed"
# Longer than a rerank takes, because the first one may not be a rerank at all:
# the model is loaded on demand and fetched from Hugging Face if the cache does
# not already hold it, so request one is a download and the rest are inference.
VMLX_RERANK_TIMEOUT = 600
