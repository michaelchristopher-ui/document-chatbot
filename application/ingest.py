from __future__ import annotations

from typing import Generator

from domain.chunking import Chunker
from domain.constants import MIN_TEXT_LEN
from domain.dedup import (
    content_digest,
    deserialize_signature,
    dedup_exact,
    document_signature,
    drop_near_duplicates,
    find_near_duplicate,
    serialize_signature,
)
from domain.errors import NoDocumentsFound
from domain.models import (
    Chunk,
    DocumentFinished,
    DocumentIndexing,
    DocumentStarted,
    EmbeddedChunk,
    IndexedDocument,
    IngestionEvent,
    IngestionFinished,
    IngestionOutcome,
    IngestionReport,
    IngestionStatus,
    PageRead,
    PageText,
    Passage,
)
from domain.text_normalization import Normalizer
from domain.titles import resolve as resolve_title
from domain.variants import IndexVariant
from ports.outbound import (
    DocumentCatalog,
    DocumentParser,
    DocumentRepository,
    EmbeddingModel,
    IndexLedger,
    KeywordIndex,
    OcrModel,
    VectorStore,
)

# What a generator yields on its way to a value: progress out, a result at the
# end. Read `Generator[IngestionEvent, None, T]` as "reports progress, returns T".
# A document returns three things: what became of it, the passages it put in the
# store — so the keyword index can be rebuilt without reading the store back —
# and which OCR model read it, empty when its text came from the file's own
# layer and no model was called.
Reporting = Generator[
    IngestionEvent, None, "tuple[IngestionOutcome, list[Passage], str]"
]


class IngestionService:
    """Index every document not already indexed, and leave the rest alone.

    An ingest is incremental in both directions. A document whose bytes are
    unchanged since the run that indexed it under this variant is reused: no
    parse, no OCR, no embedding call. Unchanged bytes are not quite the whole
    test, because a scanned page has no text of its own — what is stored for it
    is one vision model's reading, and another model would write it differently.
    So a document read by an OCR model other than the configured one is read
    again, and one that never needed OCR is left alone whatever that model is.
    A variant is never cleared to make room
    for another — chunks cut by one strategy and embedded by one model live in
    a collection of their own (see `domain.variants`), so changing either is a
    switch rather than a rebuild, and changing back finds the earlier index
    intact.

    The one thing it deletes is a changed document's own chunks, which describe
    text the file no longer contains. A document that simply leaves the folder
    keeps its chunks and stays citable.
    """

    def __init__(
        self,
        documents: DocumentRepository,
        parser: DocumentParser,
        ocr: OcrModel,
        embeddings: EmbeddingModel,
        store: VectorStore,
        keyword_index: KeywordIndex,
        chunker: Chunker,
        catalog: DocumentCatalog,
        ledger: IndexLedger,
        variant: IndexVariant,
        normalizer: Normalizer,
    ):
        self._documents = documents
        self._parser = parser
        self._ocr = ocr
        self._embeddings = embeddings
        self._store = store
        self._keyword_index = keyword_index
        self._chunk = chunker
        self._catalog = catalog
        self._ledger = ledger
        self._variant = variant
        self._normalize = normalizer

    def ingest_all(self) -> Generator[IngestionEvent, None, None]:
        """Bring the index up to date with the folder, reporting progress as it goes.

        Lazy, like `AnswerQuestions.ask`: nothing is read until the events are
        drawn, so `NoDocumentsFound` and a backend failure surface from the first
        `next` rather than from the call. The last event is always
        `IngestionFinished` — a caller that wants only the result drains the
        stream and keeps the report it carries.

        An empty folder is only an error when the index is empty too. Once
        something is indexed it stays indexed, so a folder emptied after a run
        leaves a corpus there is still every reason to answer from.
        """
        refs = self._documents.list_documents()
        self._store.ensure_ready(self._embeddings.dimension())

        # Read once, and used twice: it is what the keyword index is rebuilt
        # from at the end, and until then it is what the ledger is checked
        # against. The ledger describes a store rather than being one, and the
        # two can disagree — a `chatbot.db` deleted between runs, or the
        # in-memory backend, leaves it describing chunks that are not there.
        stored = self._store.all_passages()
        if not refs and not stored:
            raise NoDocumentsFound(self._documents.location)

        present = {passage.source_file for passage in stored}
        known = self._ledger.entries(self._variant)

        # Grown as the run goes rather than seeded from the ledger: a document
        # about to be re-indexed must not be in here when it is, or it would be
        # found to be a near-duplicate of itself.
        signatures: dict = {}
        outcomes: list[IngestionOutcome] = []
        for index, ref in enumerate(refs, start=1):
            data = ref.read()
            digest = content_digest(data)
            entry = known.get(ref.name)

            if _reusable(entry, digest, present, self._ocr.model_id):
                outcome = entry.outcome
                # No page count: nothing below takes long enough for a fraction
                # within the document to mean anything.
                yield DocumentStarted(ref.name, outcome.title, index, len(refs))
                if entry.signature:
                    signatures[ref.name] = deserialize_signature(entry.signature)
                outcomes.append(outcome)
                # Recorded again, so `last_seen` keeps meaning what it says: the
                # document is in the folder now, whenever it was indexed.
                self._catalog.record(outcome)
                yield DocumentFinished(outcome, index, len(refs), reused=True)
                continue

            # Resolved before anything can cut this short: a document that turns
            # out to be empty or a duplicate still earns its name in the catalog,
            # and whoever is watching is owed the title rather than the filename.
            title = resolve_title(self._parser.title(data), ref.name)
            yield DocumentStarted(
                ref.name, title, index, len(refs), self._parser.page_count(data)
            )

            if ref.name in present:
                # Either the file changed since the run that indexed it, or the
                # model that read its pages did. Its stored chunks are text this
                # run will not produce again, and nothing else will ever
                # supersede them: a chunk is written under a fresh id, so
                # re-indexing around them would leave both versions searchable.
                self._store.remove(ref.name)
                stored = [p for p in stored if p.source_file != ref.name]

            outcome, added, read_by = yield from self._ingest(
                ref.name, data, title, signatures
            )
            stored.extend(added)
            outcomes.append(outcome)
            # Recorded as each document lands rather than in one pass at the end,
            # so an ingest interrupted halfway leaves the catalog what it learned.
            self._catalog.record(outcome)
            self._ledger.record(
                self._variant,
                IndexedDocument(
                    outcome,
                    digest,
                    _stored_signature(signatures, ref.name),
                    ocr_model=read_by,
                ),
            )
            yield DocumentFinished(outcome, index, len(refs))

        self._keyword_index.index(stored)
        yield IngestionFinished(IngestionReport(tuple(outcomes)))

    def _ingest(
        self, filename: str, data: bytes, title: str, signatures: dict
    ) -> Reporting:
        pages, read_by = yield from self._read_pages(filename, data)
        if not pages:
            return IngestionOutcome(filename, title, IngestionStatus.EMPTY), [], read_by

        # Layer 1: document-level MinHash gate.
        signature = document_signature(" ".join(page.text for page in pages))
        duplicate_of = find_near_duplicate(signature, signatures)
        if duplicate_of:
            return (
                IngestionOutcome(
                    filename, title, IngestionStatus.DUPLICATE, duplicate_of=duplicate_of
                ),
                [],
                # Recorded even here, where nothing was stored: a duplicate is
                # only a duplicate of the text this model read, and another one
                # reading the same scan may not land on top of anything.
                read_by,
            )
        signatures[filename] = signature

        raw_chunks: list[Chunk] = []
        for page in pages:
            raw_chunks.extend(self._chunk(page.text, page.number, filename))
        if not raw_chunks:
            return IngestionOutcome(filename, title, IngestionStatus.EMPTY), [], read_by

        # Layer 2: exact-hash dedup within this document.
        chunks = dedup_exact(raw_chunks)
        # Announced before the call rather than after it: embedding a document is
        # a single request for all of its chunks, so this is the last thing said
        # before the longest silence in an ingest.
        yield DocumentIndexing(filename, len(chunks))
        vectors = self._embeddings.embed_documents([c.text for c in chunks])

        # Layer 3: cosine dedup against what is already stored — which now
        # includes documents this run never opened, so a new document is
        # measured against the whole corpus rather than against this run's.
        chunks, vectors = drop_near_duplicates(
            chunks, vectors, self._store.max_similarity(vectors)
        )
        if not chunks:
            return IngestionOutcome(filename, title, IngestionStatus.EMPTY), [], read_by

        self._store.add([EmbeddedChunk(c, v) for c, v in zip(chunks, vectors)])
        return (
            IngestionOutcome(
                filename, title, IngestionStatus.INGESTED, chunk_count=len(chunks)
            ),
            [chunk.as_passage() for chunk in chunks],
            read_by,
        )

    def _read_pages(
        self, filename: str, data: bytes
    ) -> Generator[IngestionEvent, None, "tuple[list[PageText], str]"]:
        """The document's pages, and the OCR model that read any of them.

        Empty for the model where no page needed it. Taken from the pages as
        read rather than from the ones that survive: normalization can drop a
        page to nothing, and the question the ledger asks is whether a different
        model would have been called, not whether its words lasted.
        """
        pages = []
        read_by = ""
        for page in self._parser.pages(data):
            text = page.text.strip()
            # OCR is the slow path by orders of magnitude — a model call per page
            # against milliseconds for a text layer — so it is reported as such.
            ocr = len(text) < MIN_TEXT_LEN
            if ocr:
                read_by = self._ocr.model_id
                text = self._ocr.read_page_image(page.render_image())
            if text.strip():
                pages.append(PageText(page.number, text.strip()))
            # After the page either way: this counts pages read, and one that
            # turned out to be blank still cost the reading.
            yield PageRead(filename, page.number, ocr)
        # Normalized as a document rather than a page at a time: a running head
        # is only knowable as one by repeating across pages, and OCR output goes
        # through the same chain as a text layer, having the same faults.
        return self._normalize(pages), read_by


def _reusable(
    entry: IndexedDocument | None, digest: str, present: set, ocr_model: str
) -> bool:
    """Whether `entry` still describes the file on disk, and the store agrees.

    Three questions now, and all of them have to answer yes. The digest settles
    the file: a document whose bytes have not moved would chunk and embed to
    exactly what is already stored — unless part of what is stored was never in
    the file, which is what `_same_reading` settles. `present` settles the
    store: an entry claiming chunks the store does not hold is a stale ledger,
    and trusting it would leave the document silently unsearchable. Only an
    INGESTED entry claims chunks — a duplicate or an empty document was never
    going to put any there, so there is nothing for the store to confirm.
    """
    if entry is None or entry.content_hash != digest:
        return False
    if not _same_reading(entry.ocr_model, ocr_model):
        return False
    if entry.outcome.status is not IngestionStatus.INGESTED:
        return True
    return entry.outcome.filename in present


def _same_reading(recorded: str | None, configured: str) -> bool:
    """Whether stored text would survive being read again by `configured`.

    Three answers from `recorded`, and they are not the same question asked
    three ways:

    - Empty — no page needed a model, so every word came from the file itself
      and no choice of OCR model can reach it. Reusable, always. This is most
      documents, and the reason the OCR model does not name the index the way
      the embedding model does: putting it in `IndexVariant` would re-embed a
      whole corpus over the handful of scanned pages inside it.
    - An id — the stored text is that model's reading. Reusable only while it
      is still the one configured, since another model would write those pages
      differently and the chunks cut from them differently again.
    - None — written before the ledger recorded this, so what read it cannot be
      known. Read again rather than assumed: the cost is one re-ingest of
      documents that predate the column, and the alternative is answering from
      a transcription nobody can attribute.
    """
    if recorded is None:
        return False
    if not recorded:
        return True
    return recorded == configured


def _stored_signature(signatures: dict, filename: str) -> bytes:
    """`filename`'s signature as bytes, or empty for a document that earned none.

    Only a document that passed the near-duplicate gate has one — for anything
    else there is nothing later runs would compare against.
    """
    signature = signatures.get(filename)
    return serialize_signature(signature) if signature is not None else b""
