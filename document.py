import base64
import re

import fitz  # pymupdf
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pymilvus import MilvusClient

COLLECTION_NAME = "documents"
EMBEDDING_DIM = 3072
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_TEXT_LEN = 100  # chars per page below which we OCR instead


def _ocr_page(page: fitz.Page, model_name: str, rate_limiter) -> str:
    """Render a fitz page to an image and extract text via Gemini vision."""
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    img_b64 = base64.b64encode(pix.tobytes("jpeg", jpg_quality=85)).decode()
    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        rate_limiter=rate_limiter,
        max_retries=3,
        request_options={"timeout": 120},
    )
    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": "Extract all text from this document page exactly as it appears. Return only the text, no commentary."},
    ])
    return model.invoke([msg]).content


def _chunk_text(text: str, page: int, source_file: str) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        chunk_text = text[start : start + CHUNK_SIZE].strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "page": page, "source_file": source_file})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _ensure_collection(client: MilvusClient) -> None:
    if not client.has_collection(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=EMBEDDING_DIM,
            metric_type="COSINE",
            enable_dynamic_field=True,
        )


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    client: MilvusClient,
    embeddings: GoogleGenerativeAIEmbeddings,
    model_name: str = "gemini-2.5-flash-lite",
    rate_limiter=None,
) -> int:
    _ensure_collection(client)

    chunks = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if len(text) < MIN_TEXT_LEN:
            text = _ocr_page(page, model_name, rate_limiter)
        if text.strip():
            chunks.extend(_chunk_text(text, page_num, filename))
    doc.close()

    if not chunks:
        return 0

    vectors = embeddings.embed_documents([c["text"] for c in chunks])
    client.insert(
        collection_name=COLLECTION_NAME,
        data=[{"vector": v, **c} for c, v in zip(chunks, vectors)],
    )
    return len(chunks)


def search_documents(
    query: str,
    client: MilvusClient,
    embeddings: GoogleGenerativeAIEmbeddings,
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
    pattern = r"\[Page (\d+) \| ([^\]]+)\]\n(.*?)(?=\[Page |\Z)"
    for match in re.finditer(pattern, tool_output, re.DOTALL):
        sources.append({
            "page": int(match.group(1)),
            "source_file": match.group(2).strip(),
            "text": match.group(3).strip(),
        })
    return sources
