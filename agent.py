from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from pymilvus import MilvusClient

from document import search_documents as _search_documents

SYSTEM_PROMPT = """You are a helpful document assistant. Answer questions based ONLY on the documents that have been loaded.

For every factual claim in your response, cite the source by mentioning the page number and quoting the relevant text excerpt.
If the answer cannot be found in the documents, say so clearly — do not make up information.
Always call the search_documents tool before answering any factual question."""


def build_agent(
    client: MilvusClient,
    embeddings: GoogleGenerativeAIEmbeddings,
    model_name: str = "gemini-2.5-flash-lite",
    rate_limiter=None,
):
    @tool
    def search_documents(query: str) -> str:
        """Search the loaded documents for information relevant to the query. Always call this before answering factual questions."""
        results = _search_documents(query, client, embeddings)
        if not results:
            return "No relevant content found in the documents."
        return "\n\n".join(
            f"[Page {r['page']} | {r['source_file']}]\n{r['text']}"
            for r in results
        )

    model = ChatGoogleGenerativeAI(model=model_name, temperature=0, rate_limiter=rate_limiter)
    checkpointer = InMemorySaver()

    return create_react_agent(
        model=model,
        tools=[search_documents],
        checkpointer=checkpointer,
        prompt=SYSTEM_PROMPT,
    )
