from __future__ import annotations

import base64
import hashlib
import re
from typing import Literal

import fitz  # pymupdf
import requests as _http
from datasketch import MinHash
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi

COLLECTION_NAME = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_TEXT_LEN = 100

SEMANTIC_SIMILARITY_THRESHOLD = 0.75
SEMANTIC_WINDOW_SIZE = 2

DEDUP_COSINE_THRESHOLD = 0.97
DEDUP_MINHASH_THRESHOLD = 0.85
DEDUP_MINHASH_NUM_PERM = 128

HYBRID_RETRIEVE_K = 50
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

EMBEDDING_DIM_DEFAULT = 4096

ChunkingStrategy = Literal["fixed", "recursive", "semantic"]

_HEADER_RE = re.compile(
    r'(?:^|\n)(?:[A-Z][A-Z\s\-]{3,}[A-Z]|Chapter\s+\d+|CHAPTER\s+\d+|\d+(?:\.\d+)*\s+[A-Z])(?=\s|$)',
    re.MULTILINE,
)
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_ABBREV_MAP = {
    "Mr.": "Mr\x00", "Mrs.": "Mrs\x00", "Dr.": "Dr\x00",
    "Fig.": "Fig\x00", "No.": "No\x00", "vs.": "vs\x00",
    "e.g.": "eg\x00", "i.e.": "ie\x00", "et al.": "etal\x00",
}
_ABBREV_RESTORE = {v: k for k, v in _ABBREV_MAP.items()}


def build_embeddings(model_name: str, base_url: str = LM_STUDIO_BASE_URL) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_base=base_url,
        openai_api_key="lm-studio",
    )


class LMStudioReranker:
    """Cross-encoder reranker backed by LM Studio's /v1/rerank endpoint."""

    def __init__(self, model: str, base_url: str = LM_STUDIO_BASE_URL):
        self.model = model
        self._url = base_url.rstrip("/") + "/rerank"

    def predict(self, pairs: list[list[str]]) -> list[float]:
        """Score (query, document) pairs; returns scores in input order."""
        query = pairs[0][0]
        documents = [p[1] for p in pairs]
        resp = _http.post(
            self._url,
            json={"model": self.model, "query": query, "documents": documents},
            timeout=60,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        scores = [0.0] * len(documents)
        for r in results:
            scores[r["index"]] = r["relevance_score"]
        return scores


# ── Deduplication ─────────────────────────────────────────────────────────────

def _doc_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=DEDUP_MINHASH_NUM_PERM)
    for word in text.lower().split():
        m.update(word.encode())
    return m


def _is_near_duplicate_doc(
    sig: MinHash,
    signatures: dict[str, MinHash],
) -> str | None:
    for fname, stored in signatures.items():
        if sig.jaccard(stored) >= DEDUP_MINHASH_THRESHOLD:
            return fname
    return None


def _dedup_exact(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for chunk in chunks:
        h = hashlib.sha256(chunk["text"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(chunk)
    return result


def _dedup_cosine(
    chunks: list[dict],
    vectors: list[list[float]],
    client: MilvusClient,
) -> tuple[list[dict], list[list[float]]]:
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=vectors,
            limit=1,
            output_fields=[],
        )
    except Exception:
        return chunks, vectors

    keep_chunks, keep_vectors = [], []
    for chunk, vec, hits in zip(chunks, vectors, results):
        if hits and hits[0]["distance"] >= DEDUP_COSINE_THRESHOLD:
            continue
        keep_chunks.append(chunk)
        keep_vectors.append(vec)
    return keep_chunks, keep_vectors


# ── OCR ───────────────────────────────────────────────────────────────────────

def _ocr_page(page: fitz.Page, model_name: str, base_url: str = LM_STUDIO_BASE_URL) -> str:
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    img_b64 = base64.b64encode(pix.tobytes("jpeg", jpg_quality=85)).decode()
    model = ChatOpenAI(model=model_name, base_url=base_url, api_key="lm-studio", temperature=0)
    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": "Extract all text from this document page exactly as it appears. Return only the text, no commentary."},
    ])
    return model.invoke([msg]).content


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_fixed(text: str, page: int, source_file: str) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        chunk_text = text[start : start + CHUNK_SIZE].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "page": page,
                "source_file": source_file,
                "chunking_strategy": "fixed",
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _split_on_headers(text: str) -> list[str]:
    boundaries = [m.start() for m in _HEADER_RE.finditer(text)]
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + boundaries
    boundaries.append(len(text))
    return [
        text[boundaries[i]:boundaries[i + 1]].strip()
        for i in range(len(boundaries) - 1)
        if text[boundaries[i]:boundaries[i + 1]].strip()
    ]


def _chunk_recursive(text: str, page: int, source_file: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    sections = _split_on_headers(text) or [text]
    chunks = []
    for section in sections:
        for piece in splitter.split_text(section):
            piece = piece.strip()
            if piece:
                chunks.append({
                    "text": piece,
                    "page": page,
                    "source_file": source_file,
                    "chunking_strategy": "recursive",
                })
    return chunks


def _split_sentences(text: str) -> list[str]:
    for abbrev, token in _ABBREV_MAP.items():
        text = text.replace(abbrev, token)
    sentences = _SENTENCE_RE.split(text)
    for restore_token, original in _ABBREV_RESTORE.items():
        sentences = [s.replace(restore_token, original) for s in sentences]
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _compute_breakpoints(vecs: list[list[float]]) -> set[int]:
    n = len(vecs)
    w = SEMANTIC_WINDOW_SIZE
    breakpoints = set()
    dim = len(vecs[0])
    for i in range(w, n - w):
        before = vecs[i - w : i]
        after = vecs[i : i + w]
        before_mean = [sum(v[d] for v in before) / len(before) for d in range(dim)]
        after_mean = [sum(v[d] for v in after) / len(after) for d in range(dim)]
        if _cosine_similarity(before_mean, after_mean) < SEMANTIC_SIMILARITY_THRESHOLD:
            breakpoints.add(i)
    return breakpoints


def _chunk_semantic(text: str, page: int, source_file: str, embeddings) -> list[dict]:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return _chunk_fixed(text, page, source_file)

    sentence_vecs = embeddings.embed_documents(sentences)
    breakpoints = _compute_breakpoints(sentence_vecs)

    chunks = []
    current: list[str] = []
    for i, sentence in enumerate(sentences):
        if i in breakpoints and current:
            chunk_text = " ".join(current).strip()
            if len(chunk_text) >= MIN_TEXT_LEN:
                chunks.append({
                    "text": chunk_text,
                    "page": page,
                    "source_file": source_file,
                    "chunking_strategy": "semantic",
                })
                current = [sentence]
            else:
                current.append(sentence)
        else:
            current.append(sentence)

    if current:
        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "page": page,
                "source_file": source_file,
                "chunking_strategy": "semantic",
            })

    final = []
    for chunk in chunks:
        if len(chunk["text"]) > CHUNK_SIZE * 3:
            final.extend(_chunk_fixed(chunk["text"], page, source_file))
        else:
            final.append(chunk)

    return final if final else _chunk_fixed(text, page, source_file)


def _chunk(
    text: str,
    page: int,
    source_file: str,
    strategy: ChunkingStrategy,
    embeddings=None,
) -> list[dict]:
    if strategy == "fixed":
        return _chunk_fixed(text, page, source_file)
    elif strategy == "recursive":
        return _chunk_recursive(text, page, source_file)
    elif strategy == "semantic":
        if embeddings is None:
            raise ValueError("embeddings must be provided for semantic chunking")
        return _chunk_semantic(text, page, source_file, embeddings)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")


# ── Milvus ────────────────────────────────────────────────────────────────────

def _ensure_collection(client: MilvusClient, dimension: int) -> None:
    if not client.has_collection(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=dimension,
            metric_type="COSINE",
            enable_dynamic_field=True,
        )


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    client: MilvusClient,
    embeddings,
    model_name: str,
    base_url: str = LM_STUDIO_BASE_URL,
    chunking_strategy: ChunkingStrategy = "recursive",
    doc_signatures: dict | None = None,
    embedding_dim: int = EMBEDDING_DIM_DEFAULT,
) -> int:
    """Ingest a PDF into the vector store.

    Returns the number of chunks inserted, 0 if the document yielded no
    content after deduplication, or -1 if the whole document was skipped
    because it is a near-duplicate of one already ingested this session.
    """
    _ensure_collection(client, embedding_dim)

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_texts: list[tuple[int, str]] = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if len(text) < MIN_TEXT_LEN:
            text = _ocr_page(page, model_name, base_url)
        if text.strip():
            page_texts.append((page_num, text.strip()))
    doc.close()

    if not page_texts:
        return 0

    # Layer 1: document-level MinHash gate (Jaccard >= 0.85 → skip)
    if doc_signatures is not None:
        full_text = " ".join(t for _, t in page_texts)
        sig = _doc_minhash(full_text)
        duplicate_of = _is_near_duplicate_doc(sig, doc_signatures)
        if duplicate_of:
            return -1
        doc_signatures[filename] = sig

    # Chunk
    raw_chunks: list[dict] = []
    for page_num, text in page_texts:
        raw_chunks.extend(_chunk(text, page_num, filename, chunking_strategy, embeddings))

    if not raw_chunks:
        return 0

    # Exact-hash dedup (catches overlapping windows within this document)
    chunks = _dedup_exact(raw_chunks)

    # Embed
    vectors = embeddings.embed_documents([c["text"] for c in chunks])

    # Layer 2: cosine similarity dedup against already-stored chunks (>= 0.97 → skip)
    chunks, vectors = _dedup_cosine(chunks, vectors, client)

    if not chunks:
        return 0

    client.insert(
        collection_name=COLLECTION_NAME,
        data=[{"vector": v, **c} for c, v in zip(chunks, vectors)],
    )
    return len(chunks)


def search_documents(
    query: str,
    client: MilvusClient,
    embeddings,
    top_k: int = 5,
) -> list[dict]:
    query_vector = embeddings.embed_query(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
        output_fields=["text", "page", "source_file"],
    )
    return [
        {
            "text": hit["entity"]["text"],
            "page": hit["entity"]["page"],
            "source_file": hit["entity"]["source_file"],
        }
        for hit in results[0]
    ]


def parse_tool_sources(tool_output: str) -> list[dict]:
    sources = []
    pattern = r"\[(\d+)\] Page (\d+) \| ([^\n]+)\n(.*?)(?=\[\d+\] Page |\Z)"
    for match in re.finditer(pattern, tool_output, re.DOTALL):
        sources.append({
            "page": int(match.group(2)),
            "source_file": match.group(3).strip(),
            "text": match.group(4).strip(),
        })
    return sources


# ── Hybrid retrieval ──────────────────────────────────────────────────────────

def fetch_all_chunks(client: MilvusClient) -> list[dict]:
    """Return every chunk stored in Milvus (used to build the BM25 index)."""
    try:
        return client.query(
            collection_name=COLLECTION_NAME,
            filter="source_file != ''",
            output_fields=["text", "page", "source_file"],
            limit=100_000,
        )
    except Exception:
        return []


def build_bm25_index(chunks: list[dict]) -> BM25Okapi | None:
    """Build a BM25Okapi index over chunk texts. Returns None for empty corpus."""
    if not chunks:
        return None
    return BM25Okapi([c["text"].lower().split() for c in chunks])


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion over ranked lists of text-keyed candidates."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def search_hybrid(
    query: str,
    client: MilvusClient,
    embeddings,
    bm25_chunks: list[dict],
    bm25_index: BM25Okapi,
    reranker=None,
    top_k_retrieve: int = HYBRID_RETRIEVE_K,
    top_k_final: int = 5,
) -> list[dict]:
    """BM25 + dense retrieval fused with RRF, then cross-encoder reranked."""
    if not bm25_chunks or bm25_index is None:
        return search_documents(query, client, embeddings, top_k_final)

    # BM25 sparse — score all chunks, take top-K
    bm25_scores = bm25_index.get_scores(query.lower().split())
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k_retrieve]
    bm25_texts = [bm25_chunks[i]["text"] for i in top_bm25_idx]

    # Dense vector search
    query_vector = embeddings.embed_query(query)
    try:
        dense_hits = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=top_k_retrieve,
            output_fields=["text", "page", "source_file"],
        )[0]
        dense_texts = [h["entity"]["text"] for h in dense_hits]
    except Exception:
        dense_texts = []

    # RRF fusion — text is a safe key because exact-hash dedup guarantees uniqueness
    fused_texts = _rrf([bm25_texts, dense_texts])[:top_k_retrieve]

    text_to_chunk: dict[str, dict] = {c["text"]: c for c in bm25_chunks}
    candidates = [text_to_chunk[t] for t in fused_texts if t in text_to_chunk]

    if not candidates:
        return search_documents(query, client, embeddings, top_k_final)

    # Cross-encoder reranking
    if reranker is not None:
        pairs = [[query, c["text"]] for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k_final]]

    return candidates[:top_k_final]
