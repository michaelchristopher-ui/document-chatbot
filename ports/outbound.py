"""Driven ports — everything the core needs from the outside world.

Structural `Protocol`s, so an adapter satisfies one by shape and never imports
this module to implement it. A module that *depends* on a port does import it,
to say what it takes: the application services do, and so do the adapters that
hold an `LLMProvider` rather than implementing one.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Protocol, Sequence

from domain.answers import AnswerLookup, CacheStatistics
from domain.filters import MetadataFilter
from domain.interactions import Judgement, TurnRecord
from domain.models import (
    AnswerEvent,
    ChatMessage,
    DocumentRef,
    EmbeddedChunk,
    IndexedDocument,
    IngestionOutcome,
    ModelCatalog,
    ParsedPage,
    Passage,
)
from domain.variants import IndexVariant


class LLMProvider(Protocol):
    """One inference backend's whole surface, shared by the ports below it.

    Every other model-facing port here is shaped like a *capability* — embed
    this, transcribe that, score these — because that is what the core asks for.
    This one is shaped like a *backend*: the base URL, the credentials and the
    clients live here once, and each capability adapter holds one of these
    rather than building a connection of its own. So a second backend is one
    class implementing this, not one class per capability, and the four
    adapters above it stop naming a backend at all.

    Models are named per call rather than per instance because one backend
    serves them all at the same address: a single provider answers for the chat
    model, the embedding model and the reranker without three connections to
    the same server.

    Types are deliberately plain — str, bytes, floats and `ChatMessage`. A
    backend whose SDK object some framework insists on driving itself (LangGraph
    does, see `adapters.outbound.agents.langgraph`) exposes that as an extra
    method on the implementation, declared beside the adapter that needs it, so
    no framework type reaches this module.
    """

    def chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """One completion, start to finish. The reply's text, never a fragment.

        For a caller that wants the whole answer before doing anything with it —
        a judge, a transcription. Streaming belongs to `ConversationalAgent`,
        which needs a graph around it rather than a single call.
        """

    def ocr(self, model: str, image: bytes, instruction: str) -> str:
        """Transcribe an image with a vision model, under `instruction`.

        Still a chat completion underneath — no backend here serves a dedicated
        OCR route — but the encoding an image needs is backend-specific, so it
        is spelled once here rather than at every call site.
        """

    def embed_query(self, model: str, text: str) -> list[float]: ...

    def embed_documents(self, model: str, texts: Sequence[str]) -> list[list[float]]: ...

    def embedding_dimension(self, model: str) -> int:
        """Vector width, probed from the backend. Raises BackendUnavailable.

        Here rather than derived by the caller because probing means a request,
        and the provider is what holds the connection to make one.
        """

    def rerank(self, model: str, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance of each document to the query, in input order.

        Raises BackendUnavailable when the backend serves no such route, which
        is common: reranking is outside the schema most of them implement.
        """


class EmbeddingModel(Protocol):
    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def dimension(self) -> int:
        """Vector width, probed from the backend. Raises BackendUnavailable."""


class OcrModel(Protocol):
    def read_page_image(self, jpeg: bytes) -> str:
        """Transcribe a rasterised page that carried no usable text layer."""

    @property
    def model_id(self) -> str:
        """Which model this is, so a reading can be recorded under its author.

        Asked for because a transcription is not a fact about the page, it is
        one model's account of it: another model reading the same image writes
        different text, so chunks cut from OCR output are only current while
        the model behind them is the one configured. `IngestionService` records
        this against the document and re-reads when it stops matching.
        """


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance of each document to the query, in input order."""


class VectorStore(Protocol):
    """Where embedded chunks live — the only durable state ingestion writes.

    One store holds one index variant: an instance is built against a collection
    named for the embedding model and chunking strategy that fill it (see
    `domain.variants`), so a dimension it is asked to hold twice is always the
    same dimension, and two variants of the same corpus never see each other.

    Any backend may sit behind this port as long as it honours the contract
    below; `adapters.outbound.vector_stores.registry` is the one place a backend is
    chosen, and nothing above this port names one.

    - Similarity is *cosine*, reported so that 1.0 is identical and larger is
      closer. `domain.dedup` thresholds against it directly, so a backend whose
      native metric is a distance must convert.
    - `search` must set `Passage.score` to that similarity. It is what the
      interaction log records as retrieval confidence, and a backend that leaves
      it None makes every answer look unmeasured rather than unmatched.
    - Scoping by `where` must agree with `MetadataFilter.matches`: the keyword
      arm evaluates the same filter in Python and fusion mixes both results, so
      a disagreement reintroduces passages the other arm excluded.
    """

    def ensure_ready(self, dimension: int) -> None:
        """Create the index if absent. Idempotent — leaves existing data alone."""

    def reset(self) -> None:
        """Discard everything currently indexed.

        Not what ingestion does any more — an index survives the run that built
        it, and is added to rather than replaced. This remains the way to throw
        one variant away deliberately.
        """

    def add(self, embedded: Sequence[EmbeddedChunk]) -> None: ...

    def remove(self, source_file: str) -> None:
        """Drop every chunk belonging to one document. No-op if it holds none.

        The one thing ingestion deletes, and only when a file on disk no longer
        matches the chunks stored under its name: those chunks describe a
        document that no longer exists, and leaving them would mean citing text
        the file has not contained since it changed. A document that merely
        leaves the folder is not this — it stays indexed and stays citable.
        """

    def search(
        self, vector: Sequence[float], limit: int, where: MetadataFilter | None = None
    ) -> list[Passage]:
        """Nearest `limit` passages, restricted to those `where` admits."""

    def max_similarity(self, vectors: Sequence[Sequence[float]]) -> list[float]:
        """Best cosine similarity in the store for each vector; 0.0 when empty."""

    def all_passages(self) -> list[Passage]:
        """Every stored passage.

        The keyword index is rebuilt from this after each ingest, so a backend
        that cannot enumerate its full contents cannot back this port.
        """


class KeywordIndex(Protocol):
    def index(self, passages: Sequence[Passage]) -> None: ...

    def search(
        self, query: str, limit: int, where: MetadataFilter | None = None
    ) -> list[Passage]:
        """Best `limit` matches, restricted to those `where` admits."""

    @property
    def is_empty(self) -> bool: ...


class DocumentParser(Protocol):
    def pages(self, data: bytes) -> Iterator[ParsedPage]: ...

    def page_count(self, data: bytes) -> int:
        """How many pages `pages` will yield, or 0 when that is not cheaply known.

        Asked before parsing begins, so progress can be reported as a fraction
        rather than a running total. A format whose length only emerges from
        reading it returns 0 rather than reading it twice to answer.
        """

    def title(self, data: bytes) -> str:
        """The title the document carries inside it, or "" when it carries none."""


class DocumentCatalog(Protocol):
    """The standing record of every document ingestion has seen.

    Kept apart from the vector store, and from `IndexLedger`, because it spans
    both: one row per filename however many variants have indexed it, and a row
    survives a variant nothing has rebuilt. Entries are updated in place and
    never removed, so a document that has since left the folder keeps the name
    its already-answered citations were written under.
    """

    def record(self, outcome: IngestionOutcome) -> None:
        """Insert `outcome`, or update the entry already held for its filename."""

    def title_of(self, filename: str) -> str:
        """The recorded title, or "" for a document the catalog has never seen."""

    def documents(self) -> list[IngestionOutcome]:
        """Every document on record, however it was left by the run that saw it."""


class IndexLedger(Protocol):
    """Which documents each index variant already holds, and in what state.

    What lets an ingest do nothing. The vector store can say that chunks exist
    under a filename but not whether they are still the chunks that file would
    produce; this records the bytes they were made from, so an unchanged
    document is recognised before it is opened.

    Keyed by variant as well as filename because the same document under a
    different embedding model or chunking strategy is different chunks in a
    different collection. Both stay — nothing here is cleared when the
    configuration changes — so returning to a combination used before finds its
    work already done.
    """

    def entries(self, variant: IndexVariant) -> dict[str, IndexedDocument]:
        """Everything `variant` holds, by filename. Empty for one never built."""

    def record(self, variant: IndexVariant, entry: IndexedDocument) -> None:
        """Insert `entry`, or replace the one held for its filename under `variant`."""


class AnswerJudge(Protocol):
    """A second model's reading of whether an answer follows from its sources.

    Separate from `ConversationalAgent` because it is a different model doing a
    different job, usually a smaller and faster one: judging is a single pass
    over text already written, with no tools and no conversation.
    """

    def assess(self, question: str, answer: str, sources: Sequence[str]) -> Judgement:
        """Score one answer against the passages it cited.

        Raises rather than guessing when the model returns something unreadable —
        an invented score is worse than a turn left unjudged.
        """

    @property
    def model(self) -> str:
        """Which model is doing the judging, recorded alongside its verdict."""


class InteractionLog(Protocol):
    """The standing record of every question this app has answered.

    Append-only, unlike `DocumentCatalog`: a turn is something that happened
    rather than something that exists, and nothing later revises one. So there is
    no key to update and no way for two runs to disagree about a row.

    Reads are windowed because what draws them aggregates in Python — see
    `domain.statistics` for why that is the trade made here.
    """

    def record(self, turn: TurnRecord) -> None:
        """Store one answered turn, with the searches and passages it used."""

    def turns(self, limit: int) -> list[TurnRecord]:
        """The `limit` most recently recorded turns, oldest first."""

    def unjudged(self, limit: int) -> list[TurnRecord]:
        """Answered turns no judge has scored yet, newest first."""

    def record_judgement(self, turn_id: int, judgement: Judgement) -> None:
        """Attach a judgement to a turn, replacing any it already carries."""


class AnswerCache(Protocol):
    """Answers already given, found by what a question means rather than by its words.

    Absent when nothing is configured to hold them, which is the default: this
    is the one port whose whole purpose is to not be consulted, and an app
    without it answers every question the long way round.

    Scope is the implementation's business, not the caller's. An answer is only
    valid for the index and the models that produced it, so an adapter binds
    itself to one of those combinations and a caller never passes a key — the
    same reason `VectorStore` is built against one collection rather than taking
    a variant per call.

    Two things it must guarantee. A failure is a miss: a cache is an
    optimisation, and one that cannot be reached must not take the question down
    with it. And a hit must be indistinguishable to the reader from the answer
    it stands in for — same events, same order — because it *is* that answer,
    read back.
    """

    def lookup(self, question: str) -> AnswerLookup:
        """Find an answer for `question`, or the means to record a new one.

        Never raises for an unreachable backend; see `AnswerAbsent`, which is
        what an outage looks like from here.
        """

    def statistics(self) -> CacheStatistics:
        """What this cache has been doing, and the settings behind it.

        Reported through the port rather than read off an implementation,
        because the numbers worth showing are the same whatever holds the
        entries: how often a question had been asked before, how much of that
        the similarity search found rather than a string comparison, and how
        often the store could not be reached. A backend with nothing to say
        about `entries` says so with `entries_known`, rather than reporting
        zero and calling a cache it cannot see an empty one.
        """

    def forget_all(self) -> None:
        """Discard every answer held for this index.

        What an ingest calls when it indexed something. Nothing else invalidates
        on a corpus change: the documents behind an answer can change without
        the question changing, and a cached "the documents do not cover this"
        outlives the document that arrived to cover it.
        """


class PromptLibrary(Protocol):
    """Where the prompts this app sends to a model are read from.

    Every prompt here has a default compiled into `domain.constants`, and the
    default is passed in on each call rather than held by the implementation.
    Two reasons. It keeps this port from being a second place the prompts are
    written down — an implementation that held its own copy could disagree with
    the one `domain.citations` matches against. And it makes the fallback part
    of the contract instead of an implementation's private policy: a library
    that cannot answer returns `default`, so a prompt store that is down, empty
    or slow costs nothing and the app answers exactly as it did before one
    existed.

    Which is why nothing here raises and nothing returns None. There is no such
    thing as a missing prompt from a caller's point of view — only the
    registered one or the built-in one.

    Called per use rather than once at composition, so a prompt republished
    while this process is running is picked up without a restart. An
    implementation is expected to cache: `text` sits on the answer path, and a
    network round trip per turn is not what a caller is asking for.
    """

    def text(self, key: str, default: str) -> str:
        """The body registered under `key`, or `default` when there is none."""


class DocumentRepository(Protocol):
    def list_documents(self) -> list[DocumentRef]: ...

    @property
    def location(self) -> str:
        """Human-readable origin, used in error messages."""


class Retriever(Protocol):
    def retrieve(self, query: str, where: MetadataFilter | None = None) -> list[Passage]: ...


class ConversationalAgent(Protocol):
    def stream(self, thread_id: str, question: str) -> Iterable[AnswerEvent]:
        """Answer within a conversation thread, emitting events as they arrive."""


class ModelRuntime(Protocol):
    def catalog(self) -> ModelCatalog: ...

    def download(self, identifier: str) -> str | None:
        """Fetch a model. Returns an error message, or None on success."""

    def unload_others(self, keep: Sequence[str]) -> str | None:
        """Evict every resident model but `keep`. Returns an error message, or None.

        Called before loading, so the models this app is about to use are not
        competing for memory with whatever the backend was holding already.
        """

    def ensure_loaded(self, identifier: str) -> str | None:
        """Load a model and wait for it. Returns an error message, or None."""
