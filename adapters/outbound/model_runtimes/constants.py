"""Constants for `adapters.outbound.model_runtimes`."""

from __future__ import annotations

import os

# How long to wait for a model listing, on every backend that serves one. Short,
# and shared: an unreachable server reads to the setup screen as "nothing is up",
# so this is the delay before that answer rather than a budget for real work.
CATALOG_TIMEOUT = 5


# ── LM Studio ─────────────────────────────────────────────────────────────────

DEFAULT_LMS_BIN = os.path.expanduser("~/.lmstudio/bin/lms")
LLM_TYPES = ("llm", "vlm")
# The vision-capable half of `LLM_TYPES`, and the whole of what OCR can run on.
VISION_TYPES = ("vlm",)
# LM Studio has reported both spellings across versions; accept either.
EMBEDDING_TYPES = ("embedding", "embeddings")
DOWNLOAD_TIMEOUT = 3600
UNLOAD_TIMEOUT = 60
POLL_INTERVAL = 5
MAX_WAIT = 1800
