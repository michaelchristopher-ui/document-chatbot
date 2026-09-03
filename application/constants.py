"""Constants for `application`."""

from __future__ import annotations

# How wide each arm of the hybrid search casts, and how much of the fused
# ranking the model is shown. `RETRIEVE_K` is per arm: fusion needs depth to
# have anything to disagree about, and the reranker needs candidates to reorder.
RETRIEVE_K = 50
FINAL_K = 5
