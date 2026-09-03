"""Every constant the app is configured against, before the environment speaks.

`config` is the layer that reads the environment; this is what it falls back to
and what the setup screen offers. The three frozen value types are here rather
than there because they exist only to describe `BACKENDS` and the
recommendation tables above it — a `Backend` is not a thing the app builds, it
is a thing the app was told.

`ROOT_DIR` is resolved from this file, which sits beside `config.py`, so every
path below still points at the same places it always did.

Backend names are spelled here *and* in `adapters.outbound.llm_providers.constants`,
deliberately: the two are matched by value and neither module imports the other,
which is what keeps the inner layer free of any knowledge that adapters exist.
See `adapters.outbound.llm_providers.registry`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from domain.models import ChunkingStrategy

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class RecommendedModel:
    """A model the setup screen preselects, and fetches when it is absent.

    The two ids differ when a backend renames what it downloads. LM Studio does:
    `catalog_id` is what `/api/v0/models` reports once installed (and what
    `lms load` expects), `download_id` is what `lms get` accepts — either a hub
    identifier or a full Hugging Face URL for models the hub does not carry. A
    backend that loads Hugging Face ids directly uses one string for both.
    """

    catalog_id: str
    download_id: str


@dataclass(frozen=True)
class Recommendations:
    """What the setup screen preselects, for one backend.

    Per backend because a recommendation is only ever a recommendation *for a
    server*: LM Studio runs GGUF through llama.cpp, vMLX runs MLX weights on
    Apple Silicon, and neither loads what the other is pointed at. Offering one
    list to both would recommend a download that cannot be used.
    """

    chat: RecommendedModel
    ocr: RecommendedModel
    embed: RecommendedModel
    # A plain id rather than a `RecommendedModel`: a reranker is named on the
    # request that uses it rather than loaded ahead of time, so there is nothing
    # to fetch first and no installed id to match against. Empty means the
    # backend serves no `/rerank` route, and the retriever keeps its RRF order.
    rerank: str = ""

    @property
    def downloadable(self) -> tuple[RecommendedModel, ...]:
        """The distinct models to offer to fetch — OCR is usually the chat model."""
        distinct: dict[str, RecommendedModel] = {}
        for model in (self.chat, self.ocr, self.embed):
            distinct.setdefault(model.catalog_id, model)
        return tuple(distinct.values())


@dataclass(frozen=True)
class Backend:
    """One inference server this app can be pointed at.

    Held here, and built in `adapters.outbound.llm_providers.registry` under the same
    keys — the two are matched by value, and neither module imports the other,
    the arrangement `VECTOR_BACKEND` and `vector_stores` are already in.
    """

    label: str
    base_url: str
    recommends: Recommendations
    # What to tell someone whose server is not answering. Backend-specific
    # because the fix is: LM Studio's server is a switch inside the app, vMLX's
    # is the command that starts it, and the models it will serve are chosen
    # there rather than afterwards.
    start_hint: str
    # Whether the backend can be asked to fetch a model it does not hold. LM
    # Studio can, through `lms get`. vMLX pulls from Hugging Face as part of
    # loading and has no separate fetch, so the download box is hidden there
    # rather than reporting a download that never happened.
    downloads: bool = True


LM_STUDIO = "lmstudio"
VMLX = "vmlx"
LLAMA_CPP = "llamacpp"

# Sized for a 12 GB machine, where every model a backend recommends is resident
# at once: a ~5 GB chat model + a sub-gigabyte embedder leaves room for the KV
# cache that long RAG prompts need.
LM_STUDIO_MODELS = Recommendations(
    chat=RecommendedModel(
        catalog_id="qwen/qwen3-vl-8b",
        download_id="qwen/qwen3-vl-8b",
    ),
    # The same model as the chat default, and a setting of its own all the same.
    # OCR is chosen separately on the setup screen, from vision models alone — a
    # text-only model asked to read a page image answers about a picture it never
    # saw. Choosing a different one does not break the budget above: the two jobs
    # never overlap, so the UI loads the OCR model for the ingest and swaps it for
    # the chat model when the ingest ends, leaving the peak at one VLM either way.
    ocr=RecommendedModel(
        catalog_id="qwen/qwen3-vl-8b",
        download_id="qwen/qwen3-vl-8b",
    ),
    # Qwen3-Embedding-0.6B would be the stronger retriever here, but neither build
    # loads on LM Studio's current engines: the GGUF fails on llama.cpp 2.13.0
    # ("Error loading model") and the MLX build is catalogued as an `llm`, so
    # /v1/embeddings refuses it. Nomic is the largest embedder that actually serves.
    embed=RecommendedModel(
        catalog_id="text-embedding-nomic-embed-text-v1.5",
        download_id="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF",
    ),
    # No reranker: LM Studio's REST server exposes no /rerank route at all (it
    # answers "Unexpected endpoint or method"), so there is nothing to offer.
    # Point `RERANK_BASE_URL` at a server that does serve one — see `Config`.
    rerank="",
)

# vMLX loads Hugging Face ids straight, so nothing here is renamed on the way in.
# Both halves of the retriever are stronger than the LM Studio defaults can be:
# an embedder chosen for quality rather than for what will load, and a reranker
# at all. The reranker is a *different model* from the embedder by necessity —
# one encodes chunks alone for the index, the other scores query and chunk
# together, and neither can do the other's job.
VMLX_MODELS = Recommendations(
    chat=RecommendedModel(
        catalog_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        download_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
    ),
    ocr=RecommendedModel(
        catalog_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        download_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
    ),
    # What vMLX's own embeddings guide calls its high-quality pick, and what its
    # embedding path can load: mlx-embeddings covers BERT, XLM-RoBERTa and
    # ModernBERT architectures, which is why Qwen3-Embedding is not here either.
    embed=RecommendedModel(
        catalog_id="mlx-community/embeddinggemma-300m-6bit",
        download_id="mlx-community/embeddinggemma-300m-6bit",
    ),
    # Loaded on the first /v1/rerank call and swapped whenever a request names a
    # different one, so it costs nothing until a question is asked. vMLX scores
    # this one through its CausalLM backend — see its `reranker.py`, which also
    # takes cross-encoders (BGE, ModernBERT) and jina-reranker-v3.
    rerank="Qwen/Qwen3-Reranker-0.6B",
)

# llama.cpp's own server, and in practice a reranker on a port of its own. One
# process holds one model — no swapping, no second load — so a llama-server
# doing chat is a llama-server that cannot also embed or rerank, and the three
# ids below are what to start *a* server with rather than a set to run at once.
# Nothing is renamed on the way in: `-hf <repo>` is both what fetches a model
# and what `/v1/models` reports afterwards.
LLAMA_CPP_MODELS = Recommendations(
    chat=RecommendedModel(
        catalog_id="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
        download_id="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
    ),
    ocr=RecommendedModel(
        catalog_id="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
        download_id="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
    ),
    embed=RecommendedModel(
        catalog_id="ggml-org/embeddinggemma-300M-GGUF",
        download_id="ggml-org/embeddinggemma-300M-GGUF",
    ),
    # The reason to point anything here. ~568M parameters and ~1.2 GB, which is
    # more than the 38M jina-reranker-v1-turbo this used to name — and worth it,
    # because the small one does not separate passages enough to reorder them
    # usefully. On the same query and passages, turbo scored -0.062 / -0.090 /
    # -0.093 and put an unrelated passage above a related one; this scored
    # -2.79 / -10.49 / -10.73 and got the order right. A reranker whose scores
    # sit inside a 0.03 band is a model call spent to keep RRF's order.
    #
    # It is a cross-encoder, so it only ever runs over the fused 50 a search
    # already narrowed to — not the corpus — and it costs nothing until a
    # question is asked. Multilingual, unlike turbo, which is English-only.
    rerank="gpustack/bge-reranker-v2-m3-GGUF",
)

BACKENDS: dict[str, Backend] = {
    LM_STUDIO: Backend(
        label="LM Studio",
        base_url="http://localhost:1234/v1",
        recommends=LM_STUDIO_MODELS,
        start_hint="start the server and enable the local API",
    ),
    VMLX: Backend(
        label="vMLX",
        # `vmlx serve` binds 0.0.0.0:8000; the desktop app's gateway is on 8080.
        base_url="http://localhost:8000/v1",
        recommends=VMLX_MODELS,
        start_hint=(
            "start it with `vmlx serve <chat-model> --embedding-model <embedder>` "
            "— it serves what that command names and nothing else"
        ),
        downloads=False,
    ),
    LLAMA_CPP: Backend(
        label="llama.cpp",
        # A port of its own, deliberately not 1234: the ordinary reason to run
        # this is to put a reranker beside an LM Studio that cannot serve one.
        base_url="http://localhost:1235/v1",
        recommends=LLAMA_CPP_MODELS,
        start_hint=(
            "start it with `llama-server -hf <repo> --rerank --port 1235` — "
            "`--rerank` is what mounts /v1/rerank, and without it the server "
            "answers 501 however good the model it loaded is"
        ),
        downloads=False,
    ),
}

DEFAULT_LLM_BACKEND = LM_STUDIO
DEFAULT_CHUNKING_STRATEGY: ChunkingStrategy = "recursive"
DEFAULT_DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")

NO_RERANKER = "none (RRF ordering only)"

# Off by default, and a separate choice from the chat model on purpose. Judging
# runs on request from the statistics page rather than in the answer path, so it
# can afford a different model — and a small fast one is the sensible pick, since
# the job is one pass over text that is already written.
DEFAULT_JUDGE_MODEL = ""
NO_JUDGE = "none (answers are not scored)"


DEFAULT_VECTOR_BACKEND = "milvus"
DEFAULT_VECTOR_URI = os.path.join(ROOT_DIR, "chatbot.db")
# A base name rather than the name itself: the embedding model and chunking
# strategy in use qualify it, so each pair fills a collection of its own and
# none of them ever has to be cleared for another. See `domain.variants`.
DEFAULT_VECTOR_COLLECTION = "documents"

# Its own file rather than a table inside the vector store's: the two have
# different lifetimes, and milvus-lite owns the schema of its database. Two
# tables live here — the document catalog and the index ledger, which is what
# lets an ingest recognise a document it has already indexed.
DEFAULT_CATALOG_URI = os.path.join(ROOT_DIR, "catalog.db")

# One row per answered question, plus the searches and passages behind it. Its
# own file again, and for a stronger reason than the catalog's: this is written
# on every answer and read by the statistics page while the next answer is being
# written, so it is the one store with a reader and a writer at the same moment.
DEFAULT_ANALYTICS_URI = os.path.join(ROOT_DIR, "analytics.db")

# Answer cache policy. The threshold is the library's own default and is
# deliberately strict — a semantic cache fails loudly when it is too loose,
# answering a question that was never asked, and merely costs a model call when
# it is too tight. An hour is short enough that anything the ingest hook misses
# corrects itself the same working day.
DEFAULT_ANSWER_CACHE_THRESHOLD = 0.92
DEFAULT_ANSWER_CACHE_TTL_SECONDS = 3600.0


# Prompt registry. Unset PROMPT_REGISTRY_URL and there is none: every prompt is
# the constant in `domain.constants`, which is what this app did before the
# registry existed and what it still does by default. Set it and each prompt is
# read from the registry instead, per key, still falling back to the constant
# for anything it has not been given.
#
# The TTL is the lag between publishing a prompt and a running app using it, and
# also what keeps a per-turn HTTP call off the answer path. See
# `adapters.outbound.prompt_libraries.constants`, which owns the reasoning; this
# is only the env-overridable default.
DEFAULT_PROMPT_REGISTRY_TTL_SECONDS = 300.0


# What an optional model's variable is set to when the answer is "none at all".
# Needed because empty already means "unset, use the recommendation", and those
# are different answers: a deployment that wants no reranker has to be able to
# say so without the recommendation filling the gap back in.
NONE = "none"
