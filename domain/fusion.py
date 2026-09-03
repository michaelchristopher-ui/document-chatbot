from __future__ import annotations

from typing import Sequence

from domain.constants import RRF_K


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]:
    """Fuse ranked lists of keys into one ranking, best first."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda key: scores[key], reverse=True)


def rank_positions(keys: Sequence[str]) -> dict[str, int]:
    """1-based position of each key's *first* appearance in a ranked list.

    Keyed the same way `reciprocal_rank_fusion` keys its inputs, so a rank stamped
    from this describes the passage fusion actually ranked. First occurrence rather
    than last: a repeated key's best position is its rank.
    """
    positions: dict[str, int] = {}
    for rank, key in enumerate(keys, 1):
        positions.setdefault(key, rank)
    return positions
