from __future__ import annotations

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from pymilvus import MilvusClient

from document import LM_STUDIO_BASE_URL
from document import search_documents as _search_documents
from document import search_hybrid as _search_hybrid

SYSTEM_PROMPT = """You are a document assistant. Answer questions exclusively from the numbered context blocks returned by the search_documents tool. You have no authority to use knowledge outside these blocks.

CONTEXT FORMAT
search_documents returns results as numbered blocks. Each block carries the page number and source filename:

  [1] Page 3 | annual-report.pdf
  The gross margin for fiscal year 2024 was 64.2%, up from 61.8%...

  [2] Page 3 | annual-report.pdf
  ...compared to the prior year. Operating expenses grew at a slower rate...

  [3] Page 11 | supplementary-notes.pdf
  Segment revenue is reported net of inter-company eliminations...

Documents are split into chunks before indexing, so a block may begin or end mid-sentence — this is expected. Multiple consecutive blocks from the same page often form one continuous passage; treat them together when both are returned. A block that appears incomplete does not mean the document lacks the information; a follow-up search with a refined query may surface adjacent chunks.

CITATION RULES
1. Always call search_documents before answering any factual question.
2. Answer only from the returned context blocks. Never use knowledge from your training data.
3. Place the citation index in square brackets immediately after every factual claim — not at the end of the paragraph:
   Correct:   "Revenue grew 12% [1] and operating costs fell 4% [2]."
   Incorrect: "Revenue grew 12% and operating costs fell 4%. [1][2]"
4. If a claim draws from multiple blocks, list every relevant index: "Both metrics improved [1][3]."
5. You may quote directly when precision matters. When synthesising across blocks, paraphrase and cite all sources used.
6. Never fabricate page numbers, filenames, or passage content. Never cite a block index that was not returned in the current tool call.

WHEN CONTEXT IS INSUFFICIENT
7. If the returned blocks contain no relevant information, say exactly:
   "The provided documents do not contain information about [topic]."
   Do not speculate, infer, or substitute general knowledge. The absence of information in the chunks means you cannot answer that part of the question.
8. If the context is partially relevant — some aspects addressed, others not — answer the supported aspects with citations, then state explicitly:
   "The documents do not contain information about [missing aspect]."
   Never silently skip the unanswered part.

FOLLOW-UP AND MULTI-PART QUESTIONS
9. For multi-part questions, call search_documents once per distinct information need when a single query would not surface all relevant chunks. Use a different, targeted query each time.
10. Do not describe your retrieval process or explain that you are calling a tool. Answer directly from the context as if it is naturally available to you."""


def build_agent(
    client: MilvusClient,
    embeddings,
    model_name: str,
    bm25_chunks: list | None = None,
    bm25_index=None,
    reranker=None,
    base_url: str = LM_STUDIO_BASE_URL,
):
    @tool
    def search_documents(query: str) -> str:
        """Search the loaded documents for information relevant to the query. Always call this before answering factual questions."""
        if bm25_index is not None:
            results = _search_hybrid(query, client, embeddings, bm25_chunks, bm25_index, reranker)
        else:
            results = _search_documents(query, client, embeddings)
        if not results:
            return "No relevant content found in the loaded documents."
        return "\n\n".join(
            f"[{i+1}] Page {r['page']} | {r['source_file']}\n{r['text']}"
            for i, r in enumerate(results)
        )

    model = ChatOpenAI(model=model_name, base_url=base_url, api_key="lm-studio", temperature=0)
    checkpointer = InMemorySaver()

    return create_react_agent(
        model=model,
        tools=[search_documents],
        checkpointer=checkpointer,
        prompt=SYSTEM_PROMPT,
    )
