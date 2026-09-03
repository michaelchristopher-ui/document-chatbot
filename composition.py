"""Composition root: the single place where ports meet concrete adapters.

Where state lives, since it is spread across the adapters wired here:

- **Source documents** — files on disk under `config.documents_dir`, listed by
  `FilesystemDocuments`. Nothing is ever written back to them.
- **Chunks, their vectors and their metadata** — the configured vector store
  (`config.vector_backend`), in a collection named for the embedding model and
  chunking strategy that fill it. One record per chunk, carrying the text
  alongside the vector, so it doubles as the corpus. Nothing here is cleared:
  each combination of model and strategy accumulates its own collection beside
  the others, and choosing one already built is a lookup rather than a rebuild.
- **The keyword index** — in-memory BM25, rebuilt on every ingest from what the
  store held plus whatever that ingest added. Never persisted, which is why an
  ingest that indexes nothing new still has something to do.
- **The index ledger** — SQLite at `config.catalog_uri`, one row per document
  per variant, holding the hash of the bytes that produced its chunks. What
  makes an ingest incremental: a document whose file has not changed is already
  in the store, and is left alone rather than re-read.
- **The document catalog** — SQLite at `config.catalog_uri` too, a second table
  beside the ledger. One row per document ever ingested, holding the title its
  citations are shown under. Spans every variant and outlives them all, so this
  is what remembers a document that has since left the folder.
- **Chat threads** — LangGraph's `InMemorySaver`, keyed by thread id, lost on
  restart.
- **Cached answers** — Redis at `config.redis_url`, when one is configured, in a
  keyspace named for the index variant and the chat model that wrote them. The
  only store here that is meant to be lost: everything in it can be recomputed
  by asking the model again, and Redis expires its own keys, which none of the
  embedded files below do. Emptied whenever an ingest indexes anything.
- **The interaction log** — SQLite at `config.analytics_uri`, one row per answered
  question with the searches, passages and tokens behind it. Append-only, and the
  only store written while someone is reading it: the statistics page can be open
  in one tab while another answers.

And one thing that is *not* state here, though it is read from outside: the
**prompts**. With `config.prompt_registry_url` set they come from the
`prompt-registry` service, per use and cached briefly, so editing one there
changes what this app sends without a restart or a deploy. Unset — the default —
they are the constants in `domain.constants`, and nothing is asked of anything.
Either way this app only ever reads them; the registry is where they are written.
"""

from __future__ import annotations

from dataclasses import dataclass

from adapters.outbound.agents.langgraph import LangGraphAgent
from adapters.outbound.answer_caches.semantic import SemanticAnswerCache
from adapters.outbound.capabilities.provider_models import (
    ProviderEmbeddings,
    ProviderOcr,
    ProviderReranker,
)
from adapters.outbound.catalogs.sqlite import SqliteDocumentCatalog
from adapters.outbound.document_parsers.pymupdf import PyMuPdfParser
from adapters.outbound.document_repos.filesystem import FilesystemDocuments
from adapters.outbound.interaction_logs.sqlite import SqliteInteractionLog
from adapters.outbound.judges.provider import ProviderJudge
from adapters.outbound.keyword_indexes.bm25 import Bm25Index
from adapters.outbound.ledgers.sqlite import SqliteIndexLedger
from adapters.outbound.llm_providers.registry import (
    create_model_runtime,
    create_provider,
)
from adapters.outbound.prompt_libraries import BuiltinPrompts, PromptRegistryLibrary
from adapters.outbound.splitters.langchain import RecursiveSplitter
from adapters.outbound.vector_stores.registry import create_vector_store
from application.analytics import RecordedChat, RecordingRetriever
from application.caching import CachedChat, InvalidatingIngest
from application.chat import ChatService
from application.ingest import IngestionService
from application.judging import JudgingService
from application.retrieval import HybridRetriever
from config import Config
from domain.chunking import build_chunker
from domain.citations import TitleLookup
from domain.interactions import TurnContext
from domain.text_normalization import build_normalizer
from domain.titles import from_filename
from domain.variants import IndexVariant
from ports.inbound import (
    AnswerQuestions,
    IngestDocuments,
    ListDocuments,
    ScoreAnswers,
    ViewCacheStatistics,
    ViewStatistics,
)
from ports.outbound import (
    AnswerCache,
    DocumentCatalog,
    ModelRuntime,
    PromptLibrary,
)


@dataclass(frozen=True)
class Application:
    """The driving ports a UI needs, and nothing else."""

    ingestion: IngestDocuments
    chat: AnswerQuestions
    documents: ListDocuments
    statistics: ViewStatistics
    # None when no judge model is configured, which is the default: scoring an
    # answer costs a second model call, so it is opted into rather than out of.
    scoring: ScoreAnswers | None
    # None when no answer cache is configured, which is also the default. The UI
    # keys a whole page off this being present rather than drawing one full of
    # zeroes, which would read as a cache that is broken rather than absent.
    cache: ViewCacheStatistics | None


def _title_lookup(catalog: DocumentCatalog) -> TitleLookup:
    """How a passage's document is named, for a citation that has to show it.

    Falls back rather than fails: a passage whose document predates the catalog
    still has to be shown to someone, and its filename is what is left to go on.
    """
    return lambda filename: catalog.title_of(filename) or from_filename(filename)


def _prompt_library(config: Config) -> PromptLibrary:
    """A registry-backed library when one is configured, the built-ins otherwise.

    The only choice made here, and it is made by whether `PROMPT_REGISTRY_URL`
    has a value. No registry of implementations the way the vector stores have
    one: there is nothing to name, because the two cases are "read them from
    there" and "use the ones in this repository".

    Note what this does *not* do: reach the registry to check it is there. The
    library falls back per key and per lookup, so an unreachable registry is
    something a running app recovers from on its own — and refusing to start
    over one would make a prompt store a hard dependency of answering, which is
    exactly what it must not be.
    """
    if not config.prompt_registry_enabled:
        return BuiltinPrompts()

    return PromptRegistryLibrary(
        config.prompt_registry_url,
        ttl_seconds=config.prompt_registry_ttl_seconds,
        api_token=config.prompt_registry_token or None,
    )


def build(config: Config) -> Application:
    # The one place a backend is chosen. Everything below takes this and a model
    # id, so the four things this app asks a model to do share one connection,
    # and pointing them all at another server is `LLM_BACKEND` and a URL.
    provider = create_provider(config.llm_backend, config.base_url)
    # Where the prompts come from. Built first because three of the adapters
    # below take it, and it is the one dependency they share that is not a model.
    prompts = _prompt_library(config)
    # Reranking alone may answer somewhere else, because it is the one capability
    # a backend is free to serve nothing for: LM Studio has no /rerank route, so
    # a reranker there means a second server beside it. Unset — the ordinary
    # case — and this is the same provider, sharing the connections it already
    # holds. See `Config.rerank_endpoint`.
    rerank_endpoint = config.rerank_endpoint
    rerank_provider = (
        provider if rerank_endpoint is None else create_provider(*rerank_endpoint)
    )

    embeddings = ProviderEmbeddings(provider, config.embed_model)
    # Which index this run is working in. The two choices that decide whether an
    # existing index can be reused are the two that name it, so pointing the app
    # at a different model or strategy points it at a different collection
    # rather than at chunks the new pair did not produce.
    variant = IndexVariant(config.embed_model, config.chunking_strategy)
    store = create_vector_store(
        config.vector_backend,
        config.vector_uri,
        variant.collection(config.vector_collection),
    )
    keyword_index = Bm25Index()
    catalog = SqliteDocumentCatalog(config.catalog_uri)
    ledger = SqliteIndexLedger(config.catalog_uri)

    # None unless REDIS_URL is set, which is how this is opted into. Built before
    # ingestion because both halves of the cache policy need it: one serves
    # answers, the other throws them away when the documents behind them move.
    answer_cache: AnswerCache | None = (
        SemanticAnswerCache(
            embeddings=embeddings,
            redis_url=config.redis_url,
            variant=variant,
            collection=config.vector_collection,
            chat_model=config.chat_model,
            threshold=config.answer_cache_threshold,
            ttl_seconds=config.answer_cache_ttl_seconds,
        )
        if config.answer_cache_enabled
        else None
    )

    ingestion = IngestionService(
        documents=FilesystemDocuments(config.documents_dir),
        parser=PyMuPdfParser(),
        # Its own model, which the setup screen picks from vision models only.
        # Equal to `chat_model` by default, and free to differ: this is the only
        # thing that calls it, and it is done calling before the first question,
        # so the UI hands the memory back when the ingest ends.
        ocr=ProviderOcr(provider, config.ocr_model, prompts),
        embeddings=embeddings,
        store=store,
        keyword_index=keyword_index,
        chunker=build_chunker(
            config.chunking_strategy,
            split=RecursiveSplitter(),
            embed=embeddings.embed_documents,
        ),
        catalog=catalog,
        ledger=ledger,
        variant=variant,
        # The default chain of cleanup steps. Pass a different sequence to
        # `build_normalizer` to add one (`DEFAULT_STEPS + (straighten_quotes,)`),
        # drop one, or reorder — see `domain.text_normalization`.
        normalizer=build_normalizer(),
    )

    retriever = HybridRetriever(
        store=store,
        embeddings=embeddings,
        keyword_index=keyword_index,
        # No reranker configured means fused RRF order is the final order.
        reranker=(
            ProviderReranker(rerank_provider, config.reranker_model)
            if config.reranker_model
            else None
        ),
    )
    log = SqliteInteractionLog(config.analytics_uri)

    chat = ChatService(
        LangGraphAgent(
            # Wrapped before the agent ever sees it, so a search the model runs
            # anywhere inside the ReAct loop is on the record.
            RecordingRetriever(retriever),
            provider,
            config.chat_model,
            title_of=_title_lookup(catalog),
            prompts=prompts,
        )
    )
    # Outside `ChatService`, not inside: recording reads the merged citations
    # that class produces.
    #
    # The cache goes *inside* the recorder, which is the ordering that matters:
    # a served answer still writes its turn to the interaction log, as one with
    # no searches and no tokens, which is an honest account of what happened.
    # Wrapped the other way round, every question answered from the cache would
    # be missing from the statistics entirely.
    recorded = RecordedChat(
        chat if answer_cache is None else CachedChat(chat, answer_cache),
        log,
        TurnContext(
            chat_model=config.chat_model,
            embed_model=config.embed_model,
            reranker_model=config.reranker_model,
            chunking_strategy=config.chunking_strategy,
            vector_backend=config.vector_backend,
        ),
    )

    # The catalog and the log each satisfy a driving port as they stand, so they
    # are handed to the UI directly rather than wrapped in a service that would
    # only forward.
    return Application(
        ingestion=(
            ingestion
            if answer_cache is None
            else InvalidatingIngest(ingestion, answer_cache)
        ),
        chat=recorded,
        documents=catalog,
        statistics=log,
        scoring=(
            JudgingService(
                log, ProviderJudge(provider, config.judge_model, prompts)
            )
            if config.judge_model
            else None
        ),
        # Handed over as it stands, like the catalog and the log above it: the
        # adapter already satisfies the driving port, and a service wrapping it
        # would only forward.
        cache=answer_cache,
    )


def build_model_runtime(backend: str, base_url: str) -> ModelRuntime:
    """Used by the setup screen, before a full Config exists."""
    return create_model_runtime(backend, base_url)
