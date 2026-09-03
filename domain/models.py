from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal, Optional, Union

ChunkingStrategy = Literal["fixed", "recursive", "semantic"]


# ── Content ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChunkMetadata:
    """Chunk-level provenance, derived during chunking.

    `index` is the chunk's ordinal within its page, in reading order. `start` and
    `end` are character offsets into that page's text, or -1 for a chunk that
    could not be located within it.
    """

    index: int = -1
    section: str = ""
    start: int = -1
    end: int = -1


@dataclass(frozen=True)
class Passage:
    """A retrieved span of a document, as shown to the model and the reader.

    `score` is the cosine similarity the vector store matched it at — 1.0 for an
    identical vector, larger being closer. None means no search scored it: it
    came from the keyword arm, or from an enumeration of the whole store, and a
    zero there would read as "no similarity" rather than "not measured".
    """

    text: str
    page: int
    source_file: str
    strategy: str = ""
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    score: Optional[float] = None
    # Which arm of a hybrid search found it, or None when nothing attributed it:
    # a chunk read straight out of the store, or a passage retrieved before arms
    # were recorded. See `RetrievalOrigin`.
    origin: Optional["RetrievalOrigin"] = None


@dataclass(frozen=True)
class RetrievalOrigin:
    """Which arm of a hybrid search found a passage, and where each arm ranked it.

    Ranks are 1-based within their own arm. None means that arm did not return the
    passage at all, which is also the only thing that says the arm missed it.

    Recorded because RRF's output order is the *fused* one: a passage the model read
    at position 1 may have been the keyword arm's 40th and the dense arm's 1st, and
    which of those it was is the whole of what tuning a hybrid retriever acts on.
    Without it, "the search found this" is one undifferentiated fact.

    `fused_rank` is where RRF put it, which differs from the rank the model finally
    saw only when a reranker reordered the candidates — that difference is the
    reranker's entire contribution.
    """

    keyword_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    fused_rank: Optional[int] = None


@dataclass(frozen=True)
class Citation:
    """A passage under the index the answer refers to it by.

    The index belongs to the answer rather than to the passage: it is assigned
    when a search first surfaces the passage and holds for the rest of that
    answer, so a number the model writes always means the same passage.

    That makes it a retrieval index, not a footnote number — it counts every
    passage a search returned, including the ones the answer ignored. Numbering
    the reader sees is `domain.citations.displayed_citations`, which renumbers
    the cited few from one.
    """

    index: int
    passage: Passage
    title: str


@dataclass(frozen=True)
class Chunk:
    """A span produced by ingestion, before it is stored."""

    text: str
    page: int
    source_file: str
    strategy: str
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    def as_passage(self) -> Passage:
        return Passage(
            text=self.text,
            page=self.page,
            source_file=self.source_file,
            strategy=self.strategy,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    vector: list[float]


@dataclass(frozen=True)
class ParsedPage:
    """One page of a source document.

    `render_image` is lazy: pages that already carry a text layer never pay the
    cost of rasterisation.
    """

    number: int
    text: str
    render_image: Callable[[], bytes]


@dataclass(frozen=True)
class PageText:
    """A page's text once it has been read, on its way to being chunked.

    What `domain.text_normalization` works on: the same page as `ParsedPage`,
    after whichever of the text layer or OCR supplied its words, and with no way
    left to go back to the image.
    """

    number: int
    text: str


@dataclass(frozen=True)
class DocumentRef:
    """A document available for ingestion, read on demand."""

    name: str
    read: Callable[[], bytes]


# ── Ingestion results ─────────────────────────────────────────────────────────

class IngestionStatus(Enum):
    INGESTED = "ingested"
    DUPLICATE = "duplicate"
    EMPTY = "empty"

    @classmethod
    def read(cls, value: str) -> "IngestionStatus":
        """A status as some store wrote it, including one this version predates.

        Anything unrecognised reads as EMPTY: a state written by a version that
        knew more than this one understates the row without inventing chunks
        that may no longer be indexed.
        """
        try:
            return cls(value)
        except ValueError:
            return cls.EMPTY


@dataclass(frozen=True)
class IngestionOutcome:
    """What became of one document, and what the catalog stores about it."""

    filename: str
    title: str
    status: IngestionStatus
    chunk_count: int = 0
    duplicate_of: str | None = None


@dataclass(frozen=True)
class IndexedDocument:
    """One document as an index variant already holds it.

    What lets a later run leave it alone. `content_hash` is over the file's
    bytes, which is what makes "the same document" a question with an answer: a
    file whose hash still matches is already chunked, embedded and stored, and
    reading it again would spend the same OCR and embedding calls arriving at
    the same chunks.

    `signature` is the document's MinHash, serialized. Near-duplicate detection
    compares a new document against every document already indexed — including
    ones this run never opens — so the comparison has to outlive the run that
    computed it. Empty for a document that never earned one, which is any that
    turned out to be empty or a duplicate itself.

    `ocr_model` is which model read the pages that had no text layer, and it
    reads three ways. An id means the stored text is that model's account of
    those pages, and only current while it stays configured. Empty means no
    page needed reading — the text came from the file's own layer, which no
    model choice can change. None means the row predates the column and the
    question cannot be answered from it, which is not the same as answering no.
    """

    outcome: IngestionOutcome
    content_hash: str
    signature: bytes = b""
    ocr_model: str | None = ""


@dataclass(frozen=True)
class IngestionReport:
    outcomes: tuple[IngestionOutcome, ...]

    @property
    def total_chunks(self) -> int:
        return sum(o.chunk_count for o in self.outcomes)


# ── Ingestion progress ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DocumentStarted:
    """Work has begun on one document, the `index`th of `total` this run.

    `pages` is how many pages the parser says the document holds, or 0 when it
    cannot say cheaply — a page count is a property of the format, and a reader
    watching this would rather see "page 4" than nothing at all.
    """

    filename: str
    title: str
    index: int
    total: int
    pages: int = 0


@dataclass(frozen=True)
class PageRead:
    """One page's words are in hand, from its text layer or from OCR.

    Emitted per page the parser yields, whether or not it turned out to carry
    text: it measures work done rather than text found. `ocr` is the difference
    between a page that took milliseconds and one that took a model call, which
    is the whole of why an ingest can appear to stall.
    """

    filename: str
    page: int
    ocr: bool = False


@dataclass(frozen=True)
class DocumentIndexing:
    """A document is read and chunked, and its `chunks` are going to the store.

    The other half of an ingest, and the half a page count says nothing about:
    embedding is one call for the whole document, so between this and
    `DocumentFinished` there is no finer progress to report — which is exactly
    why it is worth announcing. A thousand-chunk document spends minutes here
    with nothing else to show for it.
    """

    filename: str
    chunks: int


@dataclass(frozen=True)
class DocumentFinished:
    """One document is done — indexed, skipped as a duplicate, or left empty.

    `reused` says the outcome was read back rather than arrived at: the file is
    unchanged since the run that indexed it under this variant, so nothing was
    parsed and no model was called. Worth distinguishing, because it is the
    difference between a document that took two minutes and one that took none.
    """

    outcome: IngestionOutcome
    index: int
    total: int
    reused: bool = False


@dataclass(frozen=True)
class IngestionFinished:
    """The last event of every ingest, carrying the report for the whole run."""

    report: IngestionReport


IngestionEvent = Union[
    DocumentStarted, PageRead, DocumentIndexing, DocumentFinished, IngestionFinished
]


# ── Answer stream ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TextDelta:
    """An incremental piece of the assistant's answer."""

    text: str


@dataclass(frozen=True)
class SourcesFound:
    """The passages a search returned, already numbered for the answer."""

    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class TokensUsed:
    """What one call to the model cost.

    A turn is several calls — the ReAct loop asks again after every search — so
    this arrives more than once and the counts are meant to be added up. Not
    every backend reports usage; a turn that carries no such event spent an
    unknown number of tokens, which is not the same as having spent none.
    """

    prompt: int
    completion: int


@dataclass(frozen=True)
class AnswerConfidence:
    """How far an answer stands up, read off the answer and what it cited.

    The last event of a turn, because two of the three readings below need the
    finished answer: what a half-written sentence cites, or leaves out, is not
    yet a fact about the answer.

    Each reading runs 0.0 to 1.0, and each is None when nothing measured it —
    the convention `TurnRecord` already keeps, and the one that matters most
    here: a component averaged in as zero would report a weakness that was never
    observed. `score` is the weighted mean of the ones that were measured, and
    None when none of them were.

    - `retrieval` — how close the passages the answer leaned on actually were,
      scaled from the cosine similarity the vector store matched them at. The
      one reading available before the model writes a word.
    - `citation_coverage` — the share of the answer's claims carrying a marker
      that resolves to a passage some search returned. A marker pointing at a
      number nothing returned counts against it, which is the point: an invented
      citation is worse than none.
    - `completeness` — the share of the question's parts the answer visibly
      addresses. The weakest of the three, and weighted lowest: it is word
      overlap, not comprehension, so it catches a part dropped in silence rather
      than one answered badly.

    What this is not: a claim that the answer is *true*. It says the answer is
    anchored to passages that matched the question, and covers what was asked.
    Whether each claim follows from the passage under it is `Judgement`, which
    costs a second model call and is made later — see `application.judging`.

    `refusal` carries `domain.citations.refusal_kind`, and a full refusal comes
    back with no score at all: a correct "the documents do not cover this" is a
    good answer, and grading it on citations it was told not to make would
    report the one honest answer in the log as the least trustworthy.
    """

    score: Optional[float]
    retrieval: Optional[float] = None
    citation_coverage: Optional[float] = None
    completeness: Optional[float] = None
    refusal: Optional[str] = None
    # The closest any retrieved passage came, kept for a refusal — where it is
    # the whole reading — and as the number behind `retrieval` elsewhere.
    top_similarity: Optional[float] = None
    # The counts each share was taken over, so a reader is shown "3 of 4" rather
    # than 0.75 alone. Zero means nothing of that kind was found to count.
    claims: int = 0
    cited_claims: int = 0
    parts: int = 0
    addressed_parts: int = 0


AnswerEvent = Union[TextDelta, SourcesFound, TokensUsed, AnswerConfidence]


# ── Inference ─────────────────────────────────────────────────────────────────

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """One message in a completion request, as `LLMProvider.chat` takes them.

    The three roles every chat backend shares, and text only: the one call that
    needs an image is `LLMProvider.ocr`, which takes it as bytes rather than
    asking every caller to know how a backend wants a picture encoded.
    """

    role: Role
    content: str


# ── Model runtime ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelCatalog:
    """What the backend has installed, grouped by what it can be asked to do.

    `vision` is a subset of `llm` rather than a list beside it: a vision model
    answers a text prompt like any other, so it belongs in both. What makes it
    worth naming separately is the other direction — only these can read a page
    image, so OCR has to choose from here and nowhere else.
    """

    llm: tuple[str, ...] = ()
    embedding: tuple[str, ...] = ()
    vision: tuple[str, ...] = ()

    @property
    def online(self) -> bool:
        return bool(self.llm or self.embedding)

    def find(self, wanted: str) -> str | None:
        """The installed id for `wanted`, or None if it is not installed.

        LM Studio decorates ids it serves — `@q4_k_m` for a quantisation, `:2`
        for a second loaded instance — so an installed model rarely matches a
        recommendation character for character.
        """
        for option in (*self.llm, *self.embedding):
            if _base_id(option) == wanted.lower():
                return option
        return None


def _base_id(model_id: str) -> str:
    return model_id.lower().split("@")[0].split(":")[0]
