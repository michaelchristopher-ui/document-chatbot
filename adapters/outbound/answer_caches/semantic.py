"""Implements `AnswerCache` on the semantic-cache library.

The only module in this tree that imports `semantic_cache`, for the same reason
`vector_stores.milvus` is the only one that imports pymilvus: the port above it
says what an answer cache does, and nothing above the port should have to change
to put a different one behind it.

Three things are bound here and nowhere else.

**The namespace.** An answer is only valid for the index and the models that
produced it. The variant already names the embedding model and the chunking
strategy — the two that decide which chunks exist — and the chat model is added
because the same passages written up by a different model is a different answer.
Entries in different namespaces share nothing, so no threshold can bridge them:
changing any of the three points lookups at an empty keyspace rather than at
answers the new combination did not write.

**The embedder.** The app's own `EmbeddingModel`, so a question is measured by
the same model that indexed the corpus, and no second model has to be configured
or loaded. The library's `HashingEmbedder` is not used anywhere: it is lexical,
and a cache that only matches rewordings is a cache that misses the questions
worth catching.

**The failure boundary.** `BackendUnavailable` becomes the library's
`EmbeddingError`, which `CacheConfig.fail_open` turns into a miss. Left
untranslated it would escape the library entirely and fail the question — a
cache taking down the request it exists to make cheaper.
"""

from __future__ import annotations

from typing import Sequence

from semantic_cache import (
    CacheConfig,
    Hit,
    RedisStore,
    SemanticCache,
    SemanticCacheError,
)
from semantic_cache import (
    EmbeddingError as CacheEmbeddingError,
)

from adapters.outbound.answer_caches.codec import decode, encode
from domain.answers import (
    AnswerAbsent,
    AnswerFound,
    AnswerLookup,
    CacheStatistics,
)
from domain.errors import BackendUnavailable
from domain.models import AnswerEvent
from domain.variants import IndexVariant
from ports.outbound import EmbeddingModel


class _QuestionEmbedder:
    """Adapts this app's `EmbeddingModel` to the library's `Embedder` port.

    Two methods and a rename, and the rename is not the interesting part. The
    translation of `BackendUnavailable` is: it is what keeps an inference server
    that has gone away from turning every question into an exception instead of
    into a slow answer.
    """

    def __init__(self, embeddings: EmbeddingModel):
        self._embeddings = embeddings

    def embed(self, text: str) -> list[float]:
        try:
            return self._embeddings.embed_query(text)
        except BackendUnavailable as exc:
            raise CacheEmbeddingError(str(exc)) from exc

    def dimension(self) -> int:
        try:
            return self._embeddings.dimension()
        except BackendUnavailable as exc:
            raise CacheEmbeddingError(str(exc)) from exc


class SemanticAnswerCache:
    """Implements `AnswerCache` over Redis, keyed by meaning.

    Built for one index variant and one chat model; see the module docstring for
    why all three are in the namespace.
    """

    def __init__(
        self,
        *,
        embeddings: EmbeddingModel,
        redis_url: str,
        variant: IndexVariant,
        collection: str,
        chat_model: str,
        threshold: float,
        ttl_seconds: float,
    ):
        self._namespace = f"{variant.collection(collection)}::{chat_model}"
        self._cache = SemanticCache(
            embedder=_QuestionEmbedder(embeddings),
            # The store's TTL is Redis reclaiming the keyspace; the config's is
            # what may be served. Equal, so the two never disagree about how old
            # an answer is allowed to be.
            store=RedisStore(
                redis_url,
                ttl_seconds=ttl_seconds,
                dumps=_dumps,
                loads=decode,
            ),
            config=CacheConfig(
                threshold=threshold,
                ttl_seconds=ttl_seconds,
                # A cache that cannot be reached reports a miss and counts it,
                # rather than raising into a question someone is waiting on.
                fail_open=True,
            ),
        )

    def lookup(self, question: str) -> AnswerLookup:
        found = self._cache.lookup(question, namespace=self._namespace)
        if isinstance(found, Hit):
            return AnswerFound(
                events=tuple(found.entry.value), similarity=found.similarity
            )

        # The probe carries the normalisation and the embedding this lookup
        # already paid for. Closed over rather than returned, so `domain.answers`
        # never learns what a probe is.
        probe = found.probe

        def remember(events: Sequence[AnswerEvent]) -> None:
            self._cache.store(
                probe if probe is not None else question,
                tuple(events),
                namespace=self._namespace,
            )

        return AnswerAbsent(remember=remember, nearest=found.nearest)

    def forget_all(self) -> None:
        self._cache.clear(self._namespace)

    def statistics(self) -> CacheStatistics:
        """The library's own counters, plus what the store currently holds.

        The counters come straight off `SemanticCache.stats`, which is advisory
        by the library's own account: unsynchronised, and belonging to this
        process rather than to the keyspace. `entries` is the opposite — it is a
        question asked of Redis, so it sees what other processes wrote, and it
        is the one number here that can fail. Failing is reported rather than
        answered with a zero: an unreachable store and an empty cache are
        different facts and a reader has to be able to tell them apart.
        """
        counters = self._cache.stats
        try:
            entries = self._cache.backend.size(self._namespace)
            entries_known = True
        except SemanticCacheError:
            entries, entries_known = 0, False

        config = self._cache.config
        return CacheStatistics(
            lookups=counters.lookups,
            hits=counters.hits,
            exact_hits=counters.exact_hits,
            misses=counters.misses,
            stores=counters.stores,
            errors=counters.errors,
            entries=entries,
            entries_known=entries_known,
            threshold=config.threshold,
            # Never None here: the constructor above always sets one.
            ttl_seconds=config.ttl_seconds or 0.0,
        )

    @property
    def namespace(self) -> str:
        """Which keyspace this cache reads and writes. Shown on the stats page."""
        return self._namespace


def _dumps(value: object) -> str:
    """`encode` behind the signature the store's codec slot expects.

    The store hands whatever a caller stored; this one only ever stores a tuple
    of answer events, and anything else is a wiring mistake worth hearing about
    at the point it happens rather than at the point it is read back.
    """
    if not isinstance(value, tuple):
        raise TypeError(f"expected a tuple of answer events, got {type(value)}")
    return encode(value)
