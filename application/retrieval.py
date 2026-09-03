from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from application.constants import FINAL_K, RETRIEVE_K
from domain.filters import MetadataFilter
from domain.fusion import rank_positions, reciprocal_rank_fusion
from domain.models import Passage, RetrievalOrigin
from ports.outbound import EmbeddingModel, KeywordIndex, Reranker, VectorStore


class HybridRetriever:
    """Keyword + dense retrieval fused with RRF, then reranked if one is wired."""

    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingModel,
        keyword_index: KeywordIndex,
        reranker: Reranker | None = None,
        retrieve_k: int = RETRIEVE_K,
        final_k: int = FINAL_K,
    ):
        self._store = store
        self._embeddings = embeddings
        self._keyword_index = keyword_index
        self._reranker = reranker
        self._retrieve_k = retrieve_k
        self._final_k = final_k

    def retrieve(self, query: str, where: MetadataFilter | None = None) -> list[Passage]:
        # Both arms must honour the same scope, or fusion reintroduces passages
        # the other one excluded.
        if where is not None and where.is_empty:
            where = None

        if self._keyword_index.is_empty:
            return self._dense_only(self._dense(query, self._final_k, where))

        keyword = self._keyword_index.search(query, self._retrieve_k, where)
        dense = self._dense(query, self._retrieve_k, where)

        # RRF only ever emits keys drawn from its inputs, so this map resolves all of them.
        # Dense goes last deliberately: a passage both arms found resolves to the
        # dense copy, which is the one carrying a similarity score. Reverse the
        # order and every such passage is logged as unmeasured.
        by_text = {p.text: p for p in (*keyword, *dense)}
        keyword_texts = [p.text for p in keyword]
        dense_texts = [p.text for p in dense]
        fused = reciprocal_rank_fusion([keyword_texts, dense_texts])[: self._retrieve_k]

        # Where each arm had it before fusion, keyed on the text fusion itself keys
        # on — so a stamped rank describes the passage that was kept. `by_text`
        # already collapses two passages with byte-identical text from different
        # documents into one, and these maps collapse them the same way; a rank that
        # disagreed with the passage beside it would be worse than none. Layer-3
        # dedup at 0.95 cosine makes such a collision very unlikely in practice.
        keyword_ranks = rank_positions(keyword_texts)
        dense_ranks = rank_positions(dense_texts)
        fused_ranks = rank_positions(fused)
        candidates = [
            replace(
                by_text[text],
                origin=RetrievalOrigin(
                    keyword_rank=keyword_ranks.get(text),
                    dense_rank=dense_ranks.get(text),
                    fused_rank=fused_ranks.get(text),
                ),
            )
            for text in fused
            if text in by_text
        ]

        if not candidates:
            return self._dense_only(dense[: self._final_k])
        if self._reranker is None:
            return candidates[: self._final_k]

        try:
            scores = self._reranker.score(query, [c.text for c in candidates])
        except Exception:
            # Fused order is already a usable ranking; a reranker that is down
            # should cost precision, not the answer.
            return candidates[: self._final_k]

        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [passage for passage, _ in ranked[: self._final_k]]

    @staticmethod
    def _dense_only(passages: Sequence[Passage]) -> list[Passage]:
        """Stamp a result the dense arm produced alone, keeping its own ranking.

        `keyword_rank` stays None, which reads as "the keyword arm did not return this"
        — true whether the arm was empty or fusion had nothing to resolve. `fused_rank`
        stays None because no fusion happened, and a rank invented here would claim a
        ranking that was never computed.

        Stamped rather than left bare so a degraded search is still attributed: leave
        these paths alone and provenance goes NULL exactly when something went wrong,
        which is when it is most worth having.
        """
        return [
            replace(passage, origin=RetrievalOrigin(dense_rank=rank))
            for rank, passage in enumerate(passages, 1)
        ]

    def _dense(self, query: str, limit: int, where: MetadataFilter | None = None) -> list[Passage]:
        try:
            return self._store.search(self._embeddings.embed_query(query), limit, where)
        except Exception:
            # Degrade to keyword-only rather than failing the whole search.
            return []
