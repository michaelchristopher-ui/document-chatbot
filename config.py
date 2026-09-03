"""What this run is configured as, read from the environment.

Every default and every registered backend lives in `constants`; this is the
layer that lets the environment override them, and the one thing worth knowing
about how it does so is that nothing here reads a variable at import time —
`Config`'s fields are factories, so the lookup happens after `load_dotenv()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from constants import (
    BACKENDS,
    DEFAULT_ANALYTICS_URI,
    DEFAULT_ANSWER_CACHE_THRESHOLD,
    DEFAULT_ANSWER_CACHE_TTL_SECONDS,
    DEFAULT_CATALOG_URI,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_DOCUMENTS_DIR,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_LLM_BACKEND,
    DEFAULT_PROMPT_REGISTRY_TTL_SECONDS,
    DEFAULT_VECTOR_BACKEND,
    DEFAULT_VECTOR_COLLECTION,
    DEFAULT_VECTOR_URI,
    NONE,
    Backend,
    RecommendedModel,
    Recommendations,
)
from domain.models import ChunkingStrategy


def _env(name: str, fallback: str) -> str:
    """The variable, or `fallback` when it is unset *or empty*.

    Wrapped in a factory so the lookup happens when a Config is built, after
    `load_dotenv()` has run — a module-level `os.getenv` would read too early.

    Empty counts as unset because an orchestrator cannot express the difference:
    `docker compose` writes `FOO: ""` for every variable it lists that the
    operator left alone, so a bare `os.getenv` would read those as deliberate
    empty answers and hand `documents_dir` the empty string. Anything that
    genuinely wants "none at all" says so with `NONE` — see `_optional_model`.
    """
    return os.getenv(name) or fallback


def _float_env(name: str, fallback: float) -> float:
    """The variable as a float, or `fallback` when it is unset, empty or unreadable.

    Falls back rather than raises, unlike `backend`: a mistyped threshold makes
    the cache stricter or looser, which the statistics page shows, while a
    mistyped backend name silently answers from the wrong server. Only the
    second is worth refusing to start over.
    """
    try:
        return float(_env(name, str(fallback)))
    except ValueError:
        return fallback


def backend(name: str) -> Backend:
    """The backend registered under `name`.

    Raises rather than falling back: a typo in `LLM_BACKEND` that silently ran
    LM Studio would look like a working app pointed at the wrong server, and the
    first sign of it would be a model id nothing recognises.
    """
    try:
        return BACKENDS[name]
    except KeyError:
        known = ", ".join(BACKENDS)
        raise ValueError(
            f"Unknown inference backend {name!r}. Available: {known}."
        ) from None


def selected_backend() -> str:
    """Which backend this run is pointed at, per `LLM_BACKEND`."""
    name = _env("LLM_BACKEND", DEFAULT_LLM_BACKEND)
    backend(name)
    return name


def selected_base_url() -> str:
    """Where that backend is answering, per `LLM_BASE_URL`.

    Falls back to the backend's registered address, which is a localhost one:
    the app and the server beside each other is the ordinary case, and the
    variable is what moves the server to another machine. Its own setting rather
    than something derived from the backend name because the two are
    independent — a vMLX is a vMLX whether it is on this box or the Mac mini —
    and the mirror of `RERANK_BASE_URL`, which has always worked this way.
    """
    return _env("LLM_BASE_URL", "") or backend(selected_backend()).base_url


def selected_rerank_backend() -> str:
    """Which backend will be asked to rerank — the main one unless redirected.

    Separate from `selected_backend` because reranking is the one capability a
    backend may serve nothing for, and then it answers on a second server: the
    models to offer for it are that server's, not this one's.
    """
    name = _env("RERANK_BACKEND", "")
    if not name:
        return selected_backend()
    backend(name)
    return name


def _recommends() -> Recommendations:
    return backend(selected_backend()).recommends


def _recommended_reranker() -> str:
    """The reranker to preselect — from whichever backend will be asked for one.

    `selected_rerank_backend` rather than `_recommends`, and the difference is
    the whole point: read the main backend's recommendation and pointing an LM
    Studio at a llama.cpp would leave this empty, which `composition` reads as
    "no reranker" — reranking configured and silently off, the one outcome
    worth ruling out here.

    A bare `RERANK_BASE_URL` needs nothing special: that is the same kind of
    backend at another address, so its recommendation is already the main one's.
    """
    return backend(selected_rerank_backend()).recommends.rerank



def _model(name: str, recommended: RecommendedModel) -> str:
    """A required model id: the environment's, or the backend's recommendation."""
    return _env(name, "") or recommended.catalog_id


def _optional_model(name: str, recommended: str) -> str:
    """A model that may legitimately be absent — the reranker and the judge.

    `NONE` is how a deployment declines one that is recommended; unset falls
    through to the recommendation the way every other setting does.
    """
    chosen = _env(name, "")
    if not chosen:
        return recommended
    return "" if chosen.strip().lower() == NONE else chosen


def preconfigured() -> bool:
    """Whether the environment has answered everything the setup screen asks.

    True when the three models that have no sensible default are all named:
    chat, OCR and embedding. The reranker, the judge, the strategy and the
    server URL all have defaults worth falling back to, so requiring them would
    only make a deployment more verbose without making it more explicit.

    What it is for: an app started by a process manager has nobody to click
    through a setup screen, so this is what lets it come up answering. It also
    means the operator — not the app — owns the inference server, which is why
    `adapters.inbound.streamlit_ui` skips model loading and unloading when this
    is true. See `docs/DEPLOY.md`.
    """
    return all(_env(name, "") for name in ("CHAT_MODEL", "OCR_MODEL", "EMBED_MODEL"))


def recommended_download_id(backend_name: str, catalog_id: str) -> str | None:
    """What to hand the backend to install `catalog_id`, if it is a recommendation.

    None for anything else — a model the user picked from the catalog is already
    installed, so there is nothing to fetch.
    """
    for model in backend(backend_name).recommends.downloadable:
        if model.catalog_id == catalog_id:
            return model.download_id
    return None


@dataclass(frozen=True)
class Config:
    # Which inference server the models below are asked for, and where it is.
    # Both come from the environment rather than the setup screen: the screen
    # cannot list a single model until it knows which server to ask.
    llm_backend: str = field(default_factory=selected_backend)
    base_url: str = field(default_factory=selected_base_url)
    chat_model: str = field(default_factory=lambda: _model("CHAT_MODEL", _recommends().chat))
    # Reads the pages that carry no text layer, and must be vision-capable. Its
    # own field rather than the chat model reused: the two jobs are unrelated —
    # one transcribes an image, the other reasons over retrieved text — and the
    # best local model for each is rarely the same one.
    ocr_model: str = field(default_factory=lambda: _model("OCR_MODEL", _recommends().ocr))
    embed_model: str = field(default_factory=lambda: _model("EMBED_MODEL", _recommends().embed))
    # A cross-encoder, never the embedding model: the embedder encodes a chunk
    # without ever seeing the query, which is what makes it indexable, and the
    # reranker scores the two together, which is what makes it accurate. Empty
    # means the retriever keeps its RRF ordering — see `HybridRetriever`.
    reranker_model: str = field(
        default_factory=lambda: _optional_model("RERANKER_MODEL", _recommended_reranker())
    )
    # Empty means answers are recorded but never scored for faithfulness.
    judge_model: str = field(
        default_factory=lambda: _optional_model("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    )
    chunking_strategy: ChunkingStrategy = field(
        default_factory=lambda: _env("CHUNKING_STRATEGY", DEFAULT_CHUNKING_STRATEGY)
    )
    # An env override because a deployed app reads a mounted volume rather than
    # the folder that ships beside the source.
    documents_dir: str = field(
        default_factory=lambda: _env("DOCUMENTS_DIR", DEFAULT_DOCUMENTS_DIR)
    )

    # Where reranking is asked for, when that is not where everything else runs.
    # Its own setting because reranking is the one capability a backend is free
    # to serve nothing for: LM Studio has no /rerank route at all, so a reranker
    # there means a second server beside it — a vMLX on :8000 is one this app
    # already knows how to talk to. Empty means the one provider answers for
    # everything, which is the ordinary case.
    rerank_backend: str = field(default_factory=lambda: _env("RERANK_BACKEND", ""))
    rerank_base_url: str = field(default_factory=lambda: _env("RERANK_BASE_URL", ""))

    # Which vector store backs retrieval. Names come from
    # `adapters.outbound.vector_stores.registry`; `vector_uri` is read by that backend
    # alone — a file path for milvus-lite, a server URL for a remote Milvus.
    vector_backend: str = field(
        default_factory=lambda: _env("VECTOR_BACKEND", DEFAULT_VECTOR_BACKEND)
    )
    vector_uri: str = field(
        default_factory=lambda: _env("VECTOR_URI", DEFAULT_VECTOR_URI)
    )
    vector_collection: str = field(
        default_factory=lambda: _env("VECTOR_COLLECTION", DEFAULT_VECTOR_COLLECTION)
    )

    # SQLite file holding the document catalog — one row per document seen, with
    # the title citations are shown under — and the index ledger beside it.
    catalog_uri: str = field(
        default_factory=lambda: _env("CATALOG_URI", DEFAULT_CATALOG_URI)
    )

    # SQLite file holding the interaction log — one row per answered turn, with
    # what it retrieved, what it cited and what it cost. Read by the statistics
    # page; nothing else depends on it, so a missing file only means no history.
    analytics_uri: str = field(
        default_factory=lambda: _env("ANALYTICS_URI", DEFAULT_ANALYTICS_URI)
    )

    # Answer cache. Unset REDIS_URL and there is none: a question is answered
    # from the corpus every time, which is what this app did before this
    # existed. Set it — `docker-compose.yml` does — and a question close enough
    # to one already answered is served from Redis without a model call.
    #
    # Redis rather than a file beside the others because this is the one store
    # here that is worth sharing between replicas and worth losing: it holds
    # nothing that cannot be recomputed, and Redis expires its own keys, which
    # no embedded file does.
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", ""))

    # Minimum cosine similarity to serve a stored answer. Strict on purpose: too
    # loose and the cache answers a question nobody asked, which costs a wrong
    # answer, while too tight only costs a model call. Tune it up from what the
    # statistics page reports for real misses rather than guessing down.
    answer_cache_threshold: float = field(
        default_factory=lambda: _float_env(
            "ANSWER_CACHE_THRESHOLD", DEFAULT_ANSWER_CACHE_THRESHOLD
        )
    )

    # How long an answer may be served for. A backstop rather than the mechanism:
    # what actually invalidates an answer is the corpus behind it changing, and
    # `IngestionService` clears the cache when it indexes anything. This bounds
    # the drift that neither the TTL nor the ingest hook anticipated.
    answer_cache_ttl_seconds: float = field(
        default_factory=lambda: _float_env(
            "ANSWER_CACHE_TTL_SECONDS", DEFAULT_ANSWER_CACHE_TTL_SECONDS
        )
    )

    # Where the prompts are read from. Empty — the default — means they are the
    # constants in `domain.constants` and nothing is asked of anything.
    #
    # A URL rather than a flag because there is nothing to switch on: the
    # registry either has an address or it does not. And a URL and not a
    # `prompt-registry` dependency because the client this app talks to it with
    # is vendored (see `adapters.outbound.prompt_libraries.client`), so a
    # registry that is not running costs an import of nothing.
    prompt_registry_url: str = field(
        default_factory=lambda: _env("PROMPT_REGISTRY_URL", "")
    )

    # Bearer token, when the registry is behind something that wants one. The
    # registry's own API has no authentication — its README is explicit that
    # real auth belongs in front of it — so this is for that proxy, not for it.
    prompt_registry_token: str = field(
        default_factory=lambda: _env("PROMPT_REGISTRY_TOKEN", "")
    )

    # How long a resolved prompt is reused. `_float_env` rather than a strict
    # parse, for the same reason the cache thresholds are: a mistyped TTL makes
    # prompt edits show up sooner or later than intended, which is visible and
    # harmless, and is not worth refusing to start over.
    prompt_registry_ttl_seconds: float = field(
        default_factory=lambda: _float_env(
            "PROMPT_REGISTRY_TTL_SECONDS", DEFAULT_PROMPT_REGISTRY_TTL_SECONDS
        )
    )

    @property
    def prompt_registry_enabled(self) -> bool:
        """Whether prompts come from a registry, which is whether one is named."""
        return bool(self.prompt_registry_url)

    @property
    def answer_cache_enabled(self) -> bool:
        """Whether an answer cache is configured, which is whether Redis is named."""
        return bool(self.redis_url)

    @property
    def rerank_endpoint(self) -> tuple[str, str] | None:
        """Backend and address to rerank against, or None to use the main one.

        Either half on its own is enough to mean "elsewhere": a bare
        `RERANK_BASE_URL` is a second server of the same kind, a bare
        `RERANK_BACKEND` is a different kind at its own default address.
        """
        if not (self.rerank_backend or self.rerank_base_url):
            return None
        name = self.rerank_backend or self.llm_backend
        return name, self.rerank_base_url or backend(name).base_url
