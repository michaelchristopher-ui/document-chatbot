import glob
import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymilvus import MilvusClient

from agent import build_agent
from document import ingest_pdf, parse_tool_sources

load_dotenv()

st.set_page_config(page_title="Document Chatbot", page_icon="📄", layout="wide")

api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key


@st.cache_data(show_spinner=False)
def fetch_models(key: str) -> dict[str, list[str]]:
    try:
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": key},
            timeout=10,
        )
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return {
            "chat": sorted(
                m["name"] for m in models
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ),
            "embedding": sorted(
                m["name"] for m in models
                if "embedContent" in m.get("supportedGenerationMethods", [])
            ),
        }
    except Exception as exc:
        st.error(f"Could not fetch model list: {exc}")
        return {"chat": [], "embedding": []}


def _default_index(options: list[str], default: str) -> int:
    for i, name in enumerate(options):
        if default in name or name in default:
            return i
    return 0


@st.cache_resource(show_spinner="Processing documents…")
def load_resources(chat_model: str, embed_model: str) -> tuple:
    if not api_key:
        return None, None, {}, "GEMINI_API_KEY not found. Add it to .env or Streamlit secrets."

    pdf_paths = glob.glob(os.path.join(os.path.dirname(__file__), "documents", "*.pdf"))
    if not pdf_paths:
        return None, None, {}, "No PDFs found in the `documents/` folder."

    rate_limiter = InMemoryRateLimiter(requests_per_second=15 / 60)
    embeddings = GoogleGenerativeAIEmbeddings(model=embed_model)

    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    client = MilvusClient(db_path)

    ingestion_summary = {}
    for path in pdf_paths:
        filename = os.path.basename(path)
        with open(path, "rb") as f:
            count = ingest_pdf(f.read(), filename, client, embeddings, chat_model, rate_limiter)
        ingestion_summary[filename] = count

    agent = build_agent(client, embeddings, chat_model, rate_limiter)
    return agent, client, ingestion_summary, None


# ── Guard: API key ────────────────────────────────────────────────────────────
if not api_key:
    st.error("GEMINI_API_KEY not found. Add it to .env or Streamlit secrets.")
    st.stop()

# ── Setup screen (shown until user confirms model selection) ──────────────────
if "ready" not in st.session_state:
    available = fetch_models(api_key)

    st.title("📄 Document Chatbot")
    st.subheader("Choose models before loading your documents")
    st.caption("Models are fetched live from your Gemini API key.")

    col1, col2 = st.columns(2)
    with col1:
        chat_choice = st.selectbox(
            "Chat & OCR model (`generateContent`)",
            available["chat"],
            index=_default_index(available["chat"], os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")),
        )
    with col2:
        embed_choice = st.selectbox(
            "Embedding model (`embedContent`)",
            available["embedding"],
            index=_default_index(available["embedding"], os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")),
        )

    st.divider()
    if st.button("Load documents", type="primary", use_container_width=True):
        st.session_state.ready = True
        st.session_state.chat_model = chat_choice
        st.session_state.embed_model = embed_choice
        st.rerun()

    st.stop()

# ── Resources (loaded once after model selection) ─────────────────────────────
agent, _client, ingestion_summary, load_error = load_resources(
    st.session_state.chat_model,
    st.session_state.embed_model,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Document Chatbot")
    st.caption(f"**Chat:** `{st.session_state.chat_model}`")
    st.caption(f"**Embeddings:** `{st.session_state.embed_model}`")
    st.divider()

    if load_error:
        st.error(load_error)
    else:
        st.subheader("Loaded documents")
        for filename, count in ingestion_summary.items():
            st.success(f"✓ {filename} — {count} chunks")

    st.divider()
    if st.button("Clear chat", disabled=bool(load_error)):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    if st.button("Change models"):
        for key in ("ready", "chat_model", "embed_model", "messages", "thread_id"):
            st.session_state.pop(key, None)
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("Ask about your documents")

if load_error:
    st.error(load_error)
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**Page {src['page']} — {src['source_file']}**")
                    st.caption(src["text"][:400])

user_input = st.chat_input("Ask a question about your documents…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    response_text = ""
    sources = []

    with st.chat_message("assistant"):
        placeholder = st.empty()

        for chunk, _metadata in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str) and chunk.content:
                response_text += chunk.content
                placeholder.markdown(response_text + "▌")
            elif isinstance(chunk, ToolMessage):
                sources = parse_tool_sources(chunk.content)

        placeholder.markdown(response_text)

        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(f"**Page {src['page']} — {src['source_file']}**")
                    st.caption(src["text"][:400])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources,
    })
