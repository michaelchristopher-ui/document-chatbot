"""Streamlit UI — one driving adapter over the `Application` use cases."""

from __future__ import annotations

import html
import textwrap
import uuid
from functools import partial
from typing import Mapping, Sequence

import streamlit as st

from adapters.inbound import streamlit_cache as answer_cache
from adapters.inbound import streamlit_statistics as statistics
from adapters.inbound.constants import (
    CITATION_STYLE,
    CONFIDENCE_DOTS,
    CONFIG_KEYS,
    RECONFIGURE_KEY,
    SESSION_KEYS,
    STRATEGIES,
    TOOLTIP_CHARS,
    TOOLTIP_WIDTH,
    UNSCORED_DOT,
)
from composition import Application, build, build_model_runtime
from config import (
    Config,
    backend,
    preconfigured,
    recommended_download_id,
    selected_backend,
    selected_base_url,
    selected_rerank_backend,
)
from constants import (
    DEFAULT_CHUNKING_STRATEGY,
    NO_JUDGE,
    NO_RERANKER,
    Backend,
    RecommendedModel,
)
from domain.citations import displayed_citations, header, rewrite_markers
from domain.confidence import band
from domain.constants import FULL_REFUSAL
from domain.errors import BackendUnavailable, NoDocumentsFound
from domain.models import (
    AnswerConfidence,
    Citation,
    DocumentFinished,
    DocumentIndexing,
    DocumentStarted,
    IngestionFinished,
    IngestionOutcome,
    IngestionReport,
    IngestionStatus,
    ModelCatalog,
    PageRead,
    SourcesFound,
    TextDelta,
)
from ports.inbound import ListDocuments

# ── Cached edges ──────────────────────────────────────────────────────────────

def _server() -> Backend:
    """Which inference server this run is pointed at, and what it recommends.

    From the environment (`LLM_BACKEND`) rather than from this screen: the screen
    cannot offer a single model until it knows which server to ask, and every
    recommendation below is a recommendation *for a server* — see `constants.BACKENDS`.
    """
    return backend(selected_backend())


@st.cache_data(show_spinner=False, ttl=30)
def _catalog(backend_name: str, base_url: str) -> ModelCatalog:
    return build_model_runtime(backend_name, base_url).catalog()


@st.cache_resource(show_spinner=False)
def _application(
    backend_name: str,
    base_url: str,
    chat_model: str,
    ocr_model: str,
    embed_model: str,
    reranker_model: str,
    judge_model: str,
    chunking_strategy: str,
) -> Application:
    return build(Config(
        llm_backend=backend_name,
        base_url=base_url,
        chat_model=chat_model,
        ocr_model=ocr_model,
        embed_model=embed_model,
        reranker_model=reranker_model,
        judge_model=judge_model,
        chunking_strategy=chunking_strategy,
    ))


# One ingest per configuration, kept for the life of the process the way
# `st.cache_resource` keeps a value: a rerun re-executes the script but re-imports
# nothing, so a module global outlives it. Deliberately not that decorator —
# Streamlit records every element a cached function draws and replays them on
# each later cache hit, so the per-page progress line `_report_ingest` writes
# would be re-sent on every interaction for the rest of the session.
_LOADED: dict[
    tuple[str, ...],
    tuple[Application | None, IngestionReport | None, str | None],
] = {}


def _load(
    backend_name: str,
    base_url: str,
    chat_model: str,
    ocr_model: str,
    embed_model: str,
    reranker_model: str,
    judge_model: str,
    chunking_strategy: str,
) -> tuple[Application | None, IngestionReport | None, str | None]:
    """The application and its ingest — done once per configuration, then held."""
    key = (
        backend_name, base_url, chat_model, ocr_model, embed_model,
        reranker_model, judge_model, chunking_strategy,
    )
    if key in _LOADED:
        return _LOADED[key]

    app = _application(*key)
    try:
        report = _report_ingest(app)
    except NoDocumentsFound as exc:
        loaded = (None, None, f"No PDFs found in `{exc.location}`. Add one and reload.")
    except BackendUnavailable as exc:
        loaded = (None, None, (
            f"Cannot reach {exc.backend} at **{exc.endpoint}**. "
            f"Make sure the server is running and **{embed_model}** is loaded, "
            f"then reload this page.\n\n`{exc.detail}`"
        ))
    else:
        # Only on the way to a chat window. The two failures above end at an
        # error message, and swapping models for a page that will not answer
        # anything would be minutes spent on nothing.
        # The reranker is only this server's to hold when this server is the one
        # answering /rerank. Redirected, the id names a model another backend
        # loads, and asking this one to preload it would stall on a name it has
        # never heard of.
        # Skipped entirely when the environment configured this run: that means
        # a process manager started the app against a server somebody else
        # operates, and unloading models out from under it is not this app's
        # call. It is also the only safe answer for a *remote* LM Studio —
        # `LmsRuntime` drives `lms` as a subprocess, so those calls would act on
        # whatever happens to be installed beside the app rather than on the
        # server it is pointed at. See `config.preconfigured`.
        if not preconfigured():
            _swap_to_chat(
                backend_name,
                base_url,
                chat_model,
                embed_model,
                reranker_model if selected_rerank_backend() == backend_name else "",
            )
        loaded = (app, report, None)
        # Only a success is held. Both failures above are conditions that get
        # fixed *outside* this app — start the inference server, put a PDF in
        # the folder — so caching one would mean the app could not notice it
        # had been fixed: every later run would return the stale error and the
        # only way out would be restarting the process. A failure left uncached
        # costs one probe per rerun while the cause lasts, which is the price of
        # recovering on a reload.
        _LOADED[key] = loaded
    return loaded


def _swap_to_chat(
    backend_name: str,
    base_url: str,
    chat_model: str,
    embed_model: str,
    reranker_model: str,
) -> None:
    """Put the answering models up, and let the ingest's OCR model go.

    The two are never wanted at once: OCR reads pages before a chat window
    exists, and the chat model answers only after. Holding both for the life of
    the run would spend a second VLM on that gap for nothing — so the OCR model
    is chosen freely, on how well it reads, and costs a load here rather than
    the memory the answers need.

    A no-op when the two are the same model, which is the default: it is in
    `keep`, so nothing unloads it, and `ensure_loaded` finds it already up. A
    no-op again on a backend that binds its models when it starts — `VmlxRuntime`
    answers both calls with None, because there is nothing there to swap.
    """
    runtime = build_model_runtime(backend_name, base_url)
    keep = {chat_model, embed_model, reranker_model} - {""}

    # Warnings rather than errors, both of them. A chat window that opens and
    # stalls on the first question beats one that never opens — and with LM
    # Studio's just-in-time loading on, as it is by default, a request naming a
    # model that is down loads it rather than failing, so a stall is often all
    # this costs.
    with st.spinner("Releasing the models the documents needed…"):
        error = runtime.unload_others(keep)
    if error:
        st.warning(f"Could not unload the OCR model — {error}")

    for model_id in (chat_model, reranker_model):
        if not model_id:
            continue
        with st.spinner(f"Loading `{model_id}` for chat… (polling until ready)"):
            error = runtime.ensure_loaded(model_id)
        if error:
            st.warning(f"Could not load `{model_id}` — {error}")


# ── Ingestion progress ────────────────────────────────────────────────────────

def _report_ingest(app: Application) -> IngestionReport:
    """Run one ingest, showing what it is working on while it works.

    A document that has to be read is minutes of work, not seconds — a page with
    no text layer is a vision-model call — so a bare spinner leaves no way to
    tell a slow run from a stuck one. Every event the stream carries is drawn:
    the document in hand and its page against that document's length, the ones
    already finished, and a bar over the run as a whole. The panel collapses to
    its summary at the end, since the sidebar carries the same outcomes from
    then on.

    A run where nothing has changed passes through all of this in an instant,
    which is the point of it — so the documents it recognised say so rather than
    reading as work just done.
    """
    report = IngestionReport(())
    titles: dict[str, str] = {}
    started: DocumentStarted | None = None
    pages = 0
    reused = 0

    with st.status("Checking documents…", expanded=True) as status:
        finished = st.container()
        current = st.empty()
        bar = st.progress(0.0)

        try:
            for event in app.ingestion.ingest_all():
                if isinstance(event, DocumentStarted):
                    started = event
                    current.markdown(_working_on(event))
                    bar.progress(_fraction(event, page=0))
                elif isinstance(event, PageRead) and started is not None:
                    pages += 1
                    current.markdown(_working_on(started, _reading(started, event)))
                    bar.progress(_fraction(started, event.page))
                elif isinstance(event, DocumentIndexing) and started is not None:
                    current.markdown(
                        _working_on(started, f"embedding {_count(event.chunks, 'chunk')}")
                    )
                elif isinstance(event, DocumentFinished):
                    # Recorded before it is read: a duplicate is only ever a
                    # repeat of an earlier document, so the title it names is
                    # already here.
                    titles[event.outcome.filename] = event.outcome.title
                    reused += event.reused
                    finished.caption(
                        _outcome_line(event.outcome, titles, reused=event.reused)
                    )
                    if event.index == event.total:
                        # What is left after the last document, and not instant
                        # on a corpus of any size.
                        current.markdown("Building the keyword index…")
                elif isinstance(event, IngestionFinished):
                    report = event.report
        except Exception:
            # `st.status` turns itself red on the way out but keeps the label it
            # was given, which would leave a failed run still reading as one in
            # progress. Relabelled, then raised on for `_load` to render.
            status.update(label="Could not process the documents", state="error")
            raise

        current.empty()
        bar.progress(1.0)
        status.update(
            label=_summary(report, pages, reused), state="complete", expanded=False
        )

    return report


def _working_on(started: DocumentStarted, detail: str = "") -> str:
    """The line that says what ingestion is doing at this moment."""
    where = f"document {started.index} of {started.total}"
    return f"**{started.title}** — {where}" + (f" · {detail}" if detail else "")


def _reading(started: DocumentStarted, event: PageRead) -> str:
    """Which page is in hand, and — when it is slow — why."""
    page = f"page {event.page} of {started.pages}" if started.pages else f"page {event.page}"
    if not event.ocr:
        return page
    # Pages like this are why an ingest appears to stall: no text layer, so the
    # vision model reads the image — seconds rather than milliseconds.
    return f"{page} · reading the page image (OCR)"


def _fraction(started: DocumentStarted, page: int) -> float:
    """How far along the run is, counting pages within the document in hand.

    Documents are weighted equally regardless of length, which is what makes the
    bar move smoothly rather than truthfully: a page count is known per document
    but the run's total is not, since a document is only opened when reached.
    """
    if not started.total:
        return 1.0
    done = started.index - 1
    if started.pages:
        done += min(page / started.pages, 1.0)
    return min(done / started.total, 1.0)


def _summary(report: IngestionReport, pages: int, reused: int) -> str:
    """What the run came to, told as what it did rather than what it looked at.

    A run that read nothing does not claim to have processed anything, and an
    empty folder over a full index is a state worth naming: there is nothing to
    report on, and still a corpus to answer from.
    """
    documents = len(report.outcomes)
    if not documents:
        return "No documents to check — answering from the index already built"

    skipped = sum(1 for o in report.outcomes if o.status is not IngestionStatus.INGESTED)
    if reused == documents:
        summary = (
            f"{_count(documents, 'document')} already indexed · "
            f"{_count(report.total_chunks, 'chunk')}"
        )
    else:
        summary = (
            f"Processed {_count(documents - reused, 'document')} · "
            f"{_count(pages, 'page')} · {_count(report.total_chunks, 'chunk')}"
        )
        if reused:
            summary += f" · {reused} already indexed"
    return f"{summary} · {skipped} skipped" if skipped else summary


def _count(quantity: int, noun: str) -> str:
    """`quantity` of `noun`, pluralised and grouped — chunk counts reach five figures."""
    return f"{quantity:,} {noun}" if quantity == 1 else f"{quantity:,} {noun}s"


def _outcome_line(
    outcome: IngestionOutcome, titles: Mapping[str, str], reused: bool = False
) -> str:
    """What became of one document, in a line, for the panel and the sidebar both.

    `titles` maps filename to title: a duplicate names the document it repeats by
    filename, which is the key it was deduplicated on rather than anything worth
    reading.

    `reused` belongs to the run rather than to the document, so only the live
    panel passes it: the sidebar lists what is loaded, and a document is no less
    loaded for having been loaded before.
    """
    if outcome.status is IngestionStatus.INGESTED:
        mark = "↺" if reused else "✓"
        seen = " · already indexed" if reused else ""
        return f"{mark} {outcome.title} — {_count(outcome.chunk_count, 'chunk')}{seen}"
    if outcome.status is IngestionStatus.DUPLICATE:
        original = titles.get(outcome.duplicate_of, outcome.duplicate_of)
        return f"⊘ {outcome.title} — skipped (near-duplicate of {original})"
    return f"⚠ {outcome.title} — no chunks after deduplication"


# ── Setup screen ──────────────────────────────────────────────────────────────

def _options(
    installed: tuple[str, ...], recommended: RecommendedModel, catalog: ModelCatalog
) -> tuple[list[str], str]:
    """Installed models with the recommendation first, and that recommendation.

    The recommendation is offered even when it is not installed: selecting it is
    how the user asks for it, whether that means a download or a server started
    with it. When it *is* installed, the id the backend serves it under wins,
    since that is the one the backend will take back.
    """
    preferred = catalog.find(recommended.catalog_id) or recommended.catalog_id
    return [preferred, *(model for model in installed if model != preferred)], preferred


def _is_installed(model_id: str, catalog: ModelCatalog) -> bool:
    return model_id in (*catalog.llm, *catalog.embedding)


def _label(
    model_id: str, preferred: str, catalog: ModelCatalog, downloads: bool
) -> str:
    if model_id != preferred:
        return model_id
    # Only a backend that can be asked to fetch one promises to: elsewhere an
    # absent recommendation arrives by being named to the server at startup, or
    # not at all, and saying "will be downloaded" would promise the wrong thing.
    if downloads and not _is_installed(model_id, catalog):
        return f"{model_id}  ·  recommended, will be downloaded"
    return f"{model_id}  ·  recommended"


def _ensure_available(
    runtime, backend_name: str, model_id: str, catalog: ModelCatalog
) -> str | None:
    """Fetch `model_id` if it is a recommendation that is not installed yet."""
    if _is_installed(model_id, catalog):
        return None
    download_id = recommended_download_id(backend_name, model_id)
    if download_id is None:
        return None
    with st.spinner(f"Downloading `{download_id}` — this happens once and may take a while."):
        return runtime.download(download_id)


def _render_setup() -> None:
    backend_name = selected_backend()
    server = _server()

    st.title("📄 Document Chatbot")
    st.subheader(f"Configure {server.label}")

    base_url = st.text_input(f"{server.label} server URL", value=selected_base_url())
    catalog = _catalog(backend_name, base_url)

    if catalog.online:
        st.success(f"{server.label} server detected.")
    else:
        st.warning(
            f"Cannot reach {server.label} — {server.start_hint}, "
            "then this page will refresh."
        )

    st.divider()

    recommends = server.recommends
    llm_options, preferred_llm = _options(catalog.llm, recommends.chat, catalog)
    # Vision models alone: a text-only model asked to read a page image answers
    # about an image it never saw, so offering one here would fail silently, a
    # page of invented text at a time.
    ocr_options, preferred_ocr = _options(catalog.vision, recommends.ocr, catalog)
    emb_options, preferred_emb = _options(catalog.embedding, recommends.embed, catalog)
    # Its own list rather than the embedder's, and never the model chosen above:
    # an embedder encodes a chunk without seeing the question, which is what
    # makes it indexable, and a cross-encoder scores the two together, which is
    # what makes it accurate. Neither can stand in for the other. What is on
    # offer is whatever recommends one — and that is the backend reranking will
    # actually be asked of, which `RERANK_BACKEND` may point somewhere else
    # entirely — plus the cross-encoders LM Studio files under its `embeddings`
    # type rather than under `llm`.
    preferred_rerank = backend(selected_rerank_backend()).recommends.rerank
    rerank_options = [
        NO_RERANKER,
        *([preferred_rerank] if preferred_rerank else []),
        *(model for model in catalog.embedding if model != preferred_rerank),
    ]

    col1, col2 = st.columns(2)
    with col1:
        chat_model = st.selectbox(
            "Chat model",
            llm_options,
            format_func=lambda m: _label(m, preferred_llm, catalog, server.downloads),
            help="Answers questions from the passages retrieval returns.",
        )
        ocr_model = st.selectbox(
            "OCR model",
            ocr_options,
            # Follows the chat model whenever that one can read images, so the
            # ordinary run holds a single VLM. Choosing otherwise here sticks
            # until the chat model changes, which re-poses the question.
            index=ocr_options.index(chat_model) if chat_model in ocr_options else 0,
            format_func=lambda m: _label(m, preferred_ocr, catalog, server.downloads),
            help=(
                "Reads pages that carry no text layer — scans and images. Vision "
                "models only. A different model from the chat one is loaded for "
                "the documents and swapped out when they are done, so it costs a "
                "load rather than memory held all run."
            ),
        )
        reranker_model = st.selectbox(
            "Reranker model",
            rerank_options,
            index=(
                rerank_options.index(preferred_rerank)
                if preferred_rerank in rerank_options
                else 0
            ),
            help=(
                "Cross-encoder that re-orders the passages fusion returned — a "
                "different model from the embedder, and no substitute for it. "
                "Only a backend serving a /rerank route can run one: LM Studio's "
                "REST server does not, so leave this off there or point "
                "`RERANK_BASE_URL` at a server that does."
            ),
        )
    with col2:
        embed_model = st.selectbox(
            "Embedding model",
            emb_options,
            format_func=lambda m: _label(m, preferred_emb, catalog, server.downloads),
        )
        strategy = st.selectbox(
            "Chunking strategy",
            options=STRATEGIES,
            index=STRATEGIES.index(DEFAULT_CHUNKING_STRATEGY),
        )
        judge_model = st.selectbox(
            "Answer judge",
            [NO_JUDGE, *catalog.llm],
            help=(
                "Scores recorded answers for faithfulness to the passages they "
                "cited, from the Statistics page. Runs on request rather than "
                "while you wait, so a small fast model is the sensible pick — it "
                "need not be the model that answers."
            ),
        )

    if reranker_model == NO_RERANKER:
        reranker_model = ""
    if judge_model == NO_JUDGE:
        judge_model = ""

    # Hidden where a download is not a thing that can be asked for: vMLX pulls
    # from Hugging Face as it loads a model, so a button here would report a
    # fetch that never happened.
    if server.downloads:
        _render_download(backend_name, base_url)

    st.divider()
    if st.button(
        "Load models & documents",
        type="primary",
        use_container_width=True,
        disabled=not catalog.online,
    ):
        runtime = build_model_runtime(backend_name, base_url)
        # Which models this app *addresses* is the reader's choice; which models
        # the server is *holding* is not, once an environment configured this
        # run. That is the rule `_load` already follows, and reaching this screen
        # through "Change models" does not change who operates the server — so
        # in that case the selections are recorded and residency is left alone.
        # It costs little in practice: LM Studio's just-in-time loading brings a
        # named model up on first use, and a backend that binds its models at
        # startup was never going to swap them anyway.
        manages_residency = not preconfigured()
        configured = (
            {chat_model, ocr_model, embed_model, reranker_model} - {""}
            if manages_residency
            else set()
        )
        # What the ingest on the far side of this button actually calls, which
        # is not everything chosen above: nothing asks the chat or reranker
        # model anything until a question is typed, and by then the ingest is
        # over and the OCR model can have gone. `_swap_to_chat` does that
        # trade; here it means the pages are read with the memory the chat
        # model would otherwise be sitting on.
        ingesting = {ocr_model, embed_model} - {""}

        errors = []

        if not manages_residency:
            st.caption(
                "This run was configured from the environment, so models are "
                "selected here but not loaded or unloaded — whoever runs the "
                "inference server owns what is resident on it."
            )
        else:
            # Reclaim memory first: whatever else the backend is holding —
            # another app's model, or a duplicate instance left by an earlier run
            # of this one — is competing with the models below for a budget
            # `config` sizes with no slack. Advisory, since a failure here still
            # often loads fine, and nothing at all where the backend binds its
            # models at startup.
            with st.spinner(f"Unloading other models from {server.label}…"):
                unload_error = runtime.unload_others(ingesting)
            if unload_error:
                st.warning(f"Could not unload other models — {unload_error}")

        for model_id in configured:
            # Fetched now even when it is loaded later: a download is minutes
            # of network, and this screen is where a wait of that size belongs.
            error = _ensure_available(runtime, backend_name, model_id, catalog)
            if error:
                errors.append(f"`{model_id}`: {error}")
                continue
            if model_id not in ingesting:
                continue
            with st.spinner(f"Loading `{model_id}` in {server.label}… (polling until ready)"):
                error = runtime.ensure_loaded(model_id)
            if error:
                errors.append(f"`{model_id}`: {error}")

        if errors:
            for error in errors:
                st.error(error)
            st.stop()

        st.session_state.ready = True
        st.session_state.backend_url = base_url
        st.session_state.chat_model = chat_model
        st.session_state.ocr_model = ocr_model
        st.session_state.embed_model = embed_model
        st.session_state.reranker_model = reranker_model
        # Never preloaded, and never swapped in either: judging happens on
        # request from another page, and a third resident model would be
        # competing for memory throughout every chat that never asks for it.
        # The backend loads it on the first judge request instead.
        st.session_state.judge_model = judge_model
        st.session_state.chunking_strategy = strategy
        # The screen has been answered, so stop forcing it. Popped rather than
        # set False so the flag exists only while it is true, which is what
        # `_configured` and the gate in `main` both read it as.
        st.session_state.pop(RECONFIGURE_KEY, None)
        st.rerun()


def _render_download(backend_name: str, base_url: str) -> None:
    with st.expander("📥 Download another model"):
        st.caption(
            "A hub identifier (e.g. `qwen/qwen3-8b`), or a full Hugging Face URL for "
            "anything the hub does not carry — it stocks no embedding models at all."
        )
        left, right = st.columns([4, 1])
        with left:
            identifier = st.text_input(
                "Model identifier",
                label_visibility="collapsed",
                placeholder="publisher/model-name",
            )
        with right:
            clicked = st.button("Download", disabled=not identifier)

        if clicked and identifier:
            with st.spinner(f"Downloading `{identifier}`… this may take a while."):
                error = build_model_runtime(backend_name, base_url).download(identifier)
            if error:
                st.error(error)
            else:
                st.success(f"`{identifier}` downloaded — refresh the model lists above.")
                _catalog.clear()
                st.rerun()


# ── Main screen ───────────────────────────────────────────────────────────────

def _render_sidebar(
    app: Application | None, report: IngestionReport | None, load_error: str | None
) -> None:
    with st.sidebar:
        st.title("📄 Document Chatbot")
        st.caption(f"**Server:** `{st.session_state.backend_url}`")
        st.caption(f"**Chat:** `{st.session_state.chat_model}`")
        st.caption(f"**OCR:** `{st.session_state.ocr_model}`")
        st.caption(f"**Embed:** `{st.session_state.embed_model}`")
        st.caption(f"**Reranker:** `{st.session_state.reranker_model or 'off'}`")
        st.caption(f"**Judge:** `{st.session_state.judge_model or 'off'}`")
        st.caption(f"**Chunking:** `{st.session_state.chunking_strategy}`")
        st.divider()

        if load_error:
            st.error(load_error)
        elif report is not None:
            st.subheader("Loaded documents")
            titles = {o.filename: o.title for o in report.outcomes}
            for outcome in report.outcomes:
                line = _outcome_line(outcome, titles)
                if outcome.status is IngestionStatus.INGESTED:
                    st.success(line)
                else:
                    st.warning(line)

        if app is not None:
            _render_catalog(app.documents)

        st.divider()
        if st.button("Clear chat", disabled=bool(load_error)):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        if st.button("Change models"):
            for key in SESSION_KEYS:
                st.session_state.pop(key, None)
            # Set after the loop, not before: `SESSION_KEYS` is what the loop
            # clears, and this flag has to outlive it. Clearing state is enough
            # on its own only when nothing else can answer the setup screen's
            # questions — in a preconfigured run the environment can, and
            # `main` would re-seed from it and land straight back here.
            st.session_state[RECONFIGURE_KEY] = True
            st.rerun()


def _render_catalog(documents: ListDocuments) -> None:
    records = documents.documents()
    if not records:
        return
    with st.expander(f"All documents on record ({len(records)})"):
        st.caption("Every document ingested so far, earlier runs included.")
        for record in records:
            # The filename is the provenance a title deliberately hides, so it
            # stays one hover away rather than on the page.
            st.markdown(f"**{record.title}**", help=record.filename)
            st.caption(f"{record.status.value} · {record.chunk_count} chunks")


# ── Citations ─────────────────────────────────────────────────────────────────

def _tooltip(citation: Citation) -> str:
    text = citation.passage.text.strip()
    if len(text) > TOOLTIP_CHARS:
        text = text[:TOOLTIP_CHARS].rstrip() + "…"
    wrapped = "\n".join(textwrap.wrap(text, TOOLTIP_WIDTH)) or text
    return f"{header(citation)}\n\n{wrapped}"


def _as_attribute(value: str) -> str:
    """Fit `value` into a double-quoted HTML attribute, newlines and all."""
    return html.escape(value, quote=True).replace("\n", "&#10;")


def _with_citation_markers(answer: str, displayed: Mapping[int, Citation]) -> str:
    """Turn every `[n]` in the answer into a marker that shows what it cites."""

    def render(indices: Sequence[int]) -> str:
        marked = []
        for index in indices:
            citation = displayed.get(index)
            if citation is None:
                # An index no search returned. It is left as the model wrote it:
                # a marker that hovers to nothing is worse than plain text.
                marked.append(f"[{index}]")
            else:
                marked.append(
                    f'<span class="citation" title="{_as_attribute(_tooltip(citation))}">'
                    f"[{citation.index}]</span>"
                )
        return "".join(marked)

    return rewrite_markers(answer, render)


def _render_answer(answer: str, displayed: Mapping[int, Citation], target=st) -> None:
    """Write the answer to `target` — the page, or the placeholder it streamed into."""
    target.markdown(_with_citation_markers(answer, displayed), unsafe_allow_html=True)


def _render_sources(displayed: Mapping[int, Citation], retrieved: int) -> None:
    """List the passages the answer cites — not everything the searches returned."""
    if not displayed:
        # Searches returned passages and the answer leaned on none of them —
        # either the refusal rule 7 asks for, or an answer from somewhere other
        # than the documents. Worth saying either way: an answer that simply
        # arrives with no sources panel reads like any other.
        if retrieved:
            st.warning(
                f"This answer cites none of the {retrieved} retrieved "
                "passages — nothing in it is grounded in the documents."
            )
        return
    with st.expander(f"Sources ({len(displayed)})"):
        for citation in displayed.values():
            passage = citation.passage
            st.markdown(
                f"**[{citation.index}] {citation.title}** — page {passage.page}",
                help=passage.source_file,
            )
            if passage.metadata.section:
                st.caption(f"§ {passage.metadata.section}")
            st.caption(passage.text)


def _confidence_line(confidence: AnswerConfidence) -> str:
    """The badge as one line: the reading, then the three counts behind it.

    The breakdown is not decoration. The composite is a weighted mean, so two
    strong readings carry a weak one — an answer that is well grounded in half
    the question still scores highly — and the counts are what say which half a
    reader should be looking at.
    """
    if confidence.refusal == FULL_REFUSAL:
        # Not a low score: rule 7 asks for exactly this answer when the passages
        # hold nothing, and the closest match is the evidence it was right to.
        closest = (
            f" · closest passage {confidence.top_similarity:.2f}"
            if confidence.top_similarity is not None
            else ""
        )
        return f"{UNSCORED_DOT} Declined — the documents do not cover this{closest}"
    if confidence.score is None:
        return f"{UNSCORED_DOT} Confidence not measured"

    line = [f"{CONFIDENCE_DOTS[band(confidence)]} Confidence {confidence.score:.0%}"]
    if confidence.top_similarity is not None:
        line.append(f"best match {confidence.top_similarity:.2f}")
    if confidence.claims:
        line.append(f"{confidence.cited_claims} of {confidence.claims} claims cited")
    if confidence.parts:
        line.append(
            f"{confidence.addressed_parts} of {confidence.parts} "
            f"{'part' if confidence.parts == 1 else 'parts'} answered"
        )
    return " · ".join(line)


def _render_confidence(confidence: AnswerConfidence | None) -> None:
    """Show how far the answer stands up, and warn about a part it never reached.

    None for a message from before this existed — Streamlit keeps session state
    across a code reload, so the thread on screen may hold answers that were
    never scored. Nothing is drawn for those rather than a zero implying they
    scored badly.

    The unanswered-part warning is separate from the badge for the same reason
    the uncited-sources one is: a share folded into a composite is a number, and
    a part of the question that went unanswered is something the reader has to
    be told.
    """
    if confidence is None:
        return
    st.caption(_confidence_line(confidence))
    if confidence.parts > confidence.addressed_parts:
        missing = confidence.parts - confidence.addressed_parts
        st.warning(
            f"This answer appears to leave {missing} of the "
            f"{confidence.parts} parts of your question unanswered."
        )


def _render_assistant(
    answer: str,
    citations: Sequence[Citation],
    target=st,
    confidence: AnswerConfidence | None = None,
) -> None:
    """Render an answer, how far it stands up, and its sources under one numbering.

    Both citation halves read from the same map, so a marker in the text and its
    entry in the list cannot drift apart. Everything but the answer goes to the
    page even while the answer streams into a placeholder.
    """
    displayed = displayed_citations(answer, citations)
    _render_answer(answer, displayed, target)
    _render_confidence(confidence)
    _render_sources(displayed, len(citations))


def _render_chat(app: Application) -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant(
                    message["content"],
                    message.get("citations", ()),
                    confidence=message.get("confidence"),
                )
            else:
                st.markdown(message["content"])

    question = st.chat_input("Ask a question about your documents…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    answer = ""
    citations: tuple[Citation, ...] = ()
    confidence: AnswerConfidence | None = None

    with st.chat_message("assistant"):
        placeholder = st.empty()
        for event in app.chat.ask(st.session_state.thread_id, question):
            if isinstance(event, TextDelta):
                answer += event.text
                # Markers are left alone until the answer stops moving: a half
                # written `[1` is not yet a citation.
                placeholder.markdown(answer + "▌")
            elif isinstance(event, SourcesFound):
                citations = event.citations
            elif isinstance(event, AnswerConfidence):
                # Always the last event, so nothing is drawn until the loop ends
                # and the whole message is rendered once, in order.
                confidence = event
        _render_assistant(answer, citations, placeholder, confidence)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "confidence": confidence,
        }
    )


def _chat_page(app: Application) -> None:
    st.markdown(CITATION_STYLE, unsafe_allow_html=True)
    st.title("Ask about your documents")
    _render_chat(app)


def _configured() -> bool:
    """Whether this session has been all the way through the setup screen.

    Every key is checked rather than `ready` alone. Streamlit keeps session
    state across a code reload, so a session set up before a setting existed
    carries `ready` without it — and the honest response to that is to ask
    again, not to raise a missing key halfway down the page.
    """
    return "ready" in st.session_state and all(
        key in st.session_state for key in CONFIG_KEYS
    )


def _seed_from_env() -> None:
    """Answer the setup screen's questions from the environment, once.

    A `Config` built with no arguments *is* the resolved configuration — every
    field falls back from its variable to the backend's recommendation — so this
    reads one and writes it where the setup screen would have. Session state
    rather than a bypass of it, so everything downstream (`_load`, the sidebar,
    the statistics page) keeps reading exactly one source.
    """
    config = Config()
    st.session_state.backend_url = config.base_url
    st.session_state.chat_model = config.chat_model
    st.session_state.ocr_model = config.ocr_model
    st.session_state.embed_model = config.embed_model
    st.session_state.reranker_model = config.reranker_model
    st.session_state.judge_model = config.judge_model
    st.session_state.chunking_strategy = config.chunking_strategy
    st.session_state.ready = True


def main() -> None:
    st.set_page_config(page_title="Document Chatbot", page_icon="📄", layout="wide")

    if not _configured():
        # A deployed app has nobody to click through a setup screen. Configured
        # from the environment it seeds the same session state that screen would
        # have written and carries straight on; otherwise the screen still runs,
        # which is the interactive case and stays unchanged.
        #
        # Unless it was *asked* for. "Change models" clears session state, and
        # in a preconfigured run that is not enough on its own: the environment
        # answers the same questions again and this seeds straight past the
        # screen, so the button reads as broken. The flag is what separates
        # "never configured" from "configured, and the reader wants to choose
        # again" — the environment stays the default, not a lock.
        if not preconfigured() or st.session_state.get(RECONFIGURE_KEY):
            _render_setup()
            st.stop()
        _seed_from_env()

    app, report, load_error = _load(
        selected_backend(),
        st.session_state.backend_url,
        st.session_state.chat_model,
        st.session_state.ocr_model,
        st.session_state.embed_model,
        st.session_state.reranker_model,
        st.session_state.judge_model,
        st.session_state.chunking_strategy,
    )

    _render_sidebar(app, report, load_error)

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))

    if load_error or app is None:
        # Nothing to navigate between: without an Application there is no chat to
        # hold and no log to read, so this run stays a single page.
        st.title("Ask about your documents")
        st.error(load_error)
        st.stop()

    # `url_path` is passed rather than inferred: `st.Page` takes it from the
    # function's `__name__`, and a `partial` has none — leaving both pages sharing
    # the empty path, which Streamlit rejects as a duplicate.
    st.navigation(
        [
            st.Page(
                partial(_chat_page, app),
                title="Chat",
                icon="💬",
                url_path="chat",
                default=True,
            ),
            st.Page(
                partial(statistics.render, app.statistics, app.scoring),
                title="Statistics",
                icon="📊",
                url_path="statistics",
            ),
        ]
        + (
            # Only when one is configured. A page reporting zeroes for a cache
            # that does not exist reads as one that is broken, and there is no
            # setting on this screen that would explain the difference.
            []
            if app.cache is None
            else [
                st.Page(
                    partial(answer_cache.render, app.cache),
                    title="Answer cache",
                    icon="⚡",
                    url_path="cache",
                )
            ]
        )
    ).run()
