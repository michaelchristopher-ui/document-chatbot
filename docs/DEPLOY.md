# Deploying

The app and the inference server are separate processes that speak OpenAI-shaped
HTTP. That is the only structural fact here: **"on the same machine" is the case
where the hostname happens to resolve to loopback**, not a different mode. So
there is one image, one compose file, and one variable that decides the shape.

    LLM_BASE_URL=http://127.0.0.1:8000/v1        # beside the app
    LLM_BASE_URL=http://ai-01.internal:8000/v1   # on its own box

Nothing else changes. Don't branch on it.

## The three topologies

| | `LLM_BASE_URL` | Notes |
|---|---|---|
| App and inference on one Linux box, both in Docker | `http://inference:8000/v1` | Compose service name. Add the inference service yourself — it is not in `docker-compose.yml`, because what belongs there depends on your GPU. |
| App in Docker on a Mac mini, vMLX on that Mac | `http://host.docker.internal:8000/v1` | **vMLX cannot be containerised** — MLX needs the Mac's GPU directly, and Docker on macOS does not pass it through. It runs on the host under launchd; the container reaches it through the host gateway. |
| App anywhere, inference on its own machine | `http://ai-01.internal:8000/v1` | The one that makes the split worth having. Start vMLX with `--api-key` and set `VMLX_API_KEY` on the app. |

That second row is the one people get wrong. On a Mac, the *app* is the thing
that goes in a container; the model server never does.

## Server mode

An app started by a process manager has nobody to click through the setup
screen. Naming all three of `CHAT_MODEL`, `OCR_MODEL` and `EMBED_MODEL` is what
tells it so — `config.preconfigured()` — and it then seeds the same session
state that screen would have written and carries straight on to answering.

Two things follow from being in that mode, both deliberate:

- **The app stops managing model residency.** No downloads, no `unload_others`,
  no `ensure_loaded`. The operator owns the inference server. This is also the
  only safe answer for a *remote* LM Studio: `LmsRuntime` drives the `lms` CLI
  as a subprocess, so those calls would act on whatever is installed beside the
  app rather than on the server it is pointed at.
- **The models had better already be loaded.** vMLX binds what `vmlx serve`
  names, so this is automatic there. On a swapping backend, load them first.

`RERANKER_MODEL=none` and `JUDGE_MODEL=none` are how you decline a model that is
otherwise recommended — empty means "unset, use the recommendation", which is a
different answer.

## Pin `EMBED_MODEL` across every deployment that shares an index

`IndexVariant(embed_model, chunking_strategy)` names the vector collection. Two
apps pointed at the same store with different `EMBED_MODEL` strings build two
separate collections and neither can search the other's chunks — and the string
is what is compared, so `mlx-community/embeddinggemma-300m-6bit` and
`ggml-org/embeddinggemma-300M-GGUF` are two indexes of the same model.

`OCR_MODEL` wants the same care for a softer reason: it is recorded against each
document, and `IngestionService` re-reads any document whose recorded OCR model
no longer matches. Drift it and every ingest re-OCRs the corpus.

`CHAT_MODEL` is free to differ anywhere. It names nothing.

## There is no database container, and that is not an oversight

`chatbot.db`, `catalog.db` and `analytics.db` are **files**, not servers.
milvus-lite ships its engine as a bundled binary the client runs against a local
file, and the catalog, ledger and interaction log are SQLite. So the whole stack
is one container and one volume, and `docker compose up` needs nothing beside it.

That holds on every platform worth deploying to: milvus-lite publishes
`manylinux2014` wheels for both `x86_64` and `aarch64`, so an ARM box is fine.

### When you would actually add one

The trigger is **a second app process**, and nothing before it. Both of these
stores are embedded and single-writer, and three pieces of state never leave
memory at all — the BM25 keyword index, LangGraph's `InMemorySaver` threads, and
whatever a rerun is holding. Two replicas would not share any of it, and would
quietly disagree about search results rather than fail.

So you add real infrastructure when you want more than one replica, or when the
index outgrows one machine:

| Today | Then | Cost |
|---|---|---|
| milvus-lite file | Milvus server | `VECTOR_BACKEND=milvus`, `VECTOR_URI=http://milvus:19530` — **config only**, the adapter already handles both |
| SQLite catalog + ledger | Postgres | a new adapter per port; the SQLite ones stay |
| SQLite interaction log | Postgres | as above |
| `InMemorySaver` | LangGraph Postgres checkpointer | swap in `composition.build` |
| in-memory BM25 | OpenSearch, Postgres FTS, or Milvus sparse | a new adapter, and the only genuinely new code here |

Only the first row is free. The rest is adapter work against ports that already
exist, which is the reason it is a day rather than a rewrite — but it is not a
compose change, so do not reach for it until a second replica is the actual
requirement.

## Storage

Everything the app writes goes to the `/data` volume — `VECTOR_URI`,
`CATALOG_URI`, `ANALYTICS_URI`. The PDFs are mounted read-only at
`/data/documents`, because `FilesystemDocuments` never writes back.

Two names, on purpose: `DOCUMENTS_HOST_DIR` is the folder on the host to mount,
and `DOCUMENTS_DIR` is where the app looks *inside* the container, which the
image already fixes at `/data/documents`. You set the first one and leave the
second alone.

The index is rebuilt from the documents; the analytics log is not. If you keep
one thing, keep `analytics.db`.

## What this deployment is not

One container is one Streamlit process, and Streamlit holds a WebSocket per
browser session. That is fine for a team behind a VPN. It does not scale
horizontally as it stands: the BM25 keyword index is in-memory and rebuilt per
ingest, LangGraph threads are in an `InMemorySaver`, and milvus-lite is an
embedded single-process store. Two replicas would disagree with each other.
Going wider means a real Milvus, Postgres for catalog/ledger/analytics and
checkpoints, and an ingest worker off the request path.
