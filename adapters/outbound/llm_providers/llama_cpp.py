"""llama.cpp's own server behind `ports.outbound.LLMProvider`.

The one module that knows this backend exists. `llama-server` is the HTTP front
end shipped with llama.cpp (github.com/ggml-org/llama.cpp), and it speaks the
OpenAI chat and embeddings API the same way LM Studio does, so the same client
talks to both.

What earns it an adapter is `/v1/rerank`. LM Studio serves no such route — it
answers `{"error": "Unexpected endpoint or method. (POST /v1/rerank)"}` at every
spelling — and that is not a version away from changing, because llama.cpp is
*embedded* there rather than fronted by its own server: the engine is loaded as
a native module (`llm_engine.node`, beside the `libllama` dylibs under
`~/.lmstudio/extensions/backends/`) and driven over a private protocol. Recent
engine builds even ship the `llama-server` binary, unused, in that same folder.
So the route llama.cpp implements is real and simply unreachable from port 1234;
reaching it means running that binary, which makes this a second backend at a
second address rather than a second route on the first.

Started for reranking alone, beside whatever answers everything else:

    llama-server -hf ggml-org/jina-reranker-v1-turbo-en-GGUF --rerank --port 1235

then `RERANK_BACKEND=llamacpp` and `RERANK_BASE_URL=http://localhost:1235/v1` —
see `Config.rerank_endpoint`, which is what wires one backend's reranker under
another's chat and embeddings.

Where it departs from the OpenAI schema, and from the other backends here:

- `/v1/rerank` is Cohere/Jina-shaped rather than OpenAI, so it is posted by
  hand. `top_n` is deliberately not sent: omitted, the server scores every
  document, and this port promises a score per document rather than a shortlist.
- Failures arrive as a 4xx/5xx with an `{"error": {"message": ...}}` body, where
  LM Studio answers 200 with `{"error": ...}` and vMLX answers FastAPI's
  `{"detail": ...}`. So the status is what is checked here, and the body only
  decides what the error says. The one worth recognising is the 501: a server
  holding a perfectly good reranker but started without `--rerank` never mounts
  the route, and says exactly that.
- Scores are the cross-encoder's raw logits — unbounded, and negative more often
  than not. Sorting on them is all they support, which is all `HybridRetriever`
  does; they are not probabilities and there is no cutoff to compare one to.
- Authentication is off until the server is started with `--api-key`, and is a
  bearer token when it is on. `LLAMA_API_KEY` is where this reads one, the same
  name llama-server itself reads.

One process serves one model, fixed at launch — `adapters.outbound.model_runtimes.llama_cpp`
covers what that costs the setup screen. That is also why nothing here binds a
model name for reranking: the request may carry one, and the server ignores it
and echoes it back.
"""

from __future__ import annotations

import base64
from typing import Sequence

import requests
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from adapters.outbound.llm_providers.constants import (
    LLAMA_CPP_API_KEY,
    LLAMA_CPP_LABEL,
    LLAMA_CPP_RERANK_TIMEOUT,
    PROBE_TEXT,
)
from domain.errors import BackendUnavailable
from domain.models import ChatMessage


class LlamaCppProvider:
    """Implements `LLMProvider`.

    Clients are built on first use and kept, keyed by the configuration they
    were built with. Every one of them opens an HTTP connection pool, and a
    judge that made a fresh client per answer would open one per answer.
    """

    def __init__(self, base_url: str, api_key: str = LLAMA_CPP_API_KEY):
        self._base_url = base_url
        self._api_key = api_key
        self._chat_clients: dict[tuple, ChatOpenAI] = {}
        self._embed_clients: dict[str, OpenAIEmbeddings] = {}
        self._dimensions: dict[str, int] = {}

    @property
    def base_url(self) -> str:
        """Where this provider is pointed, for an error that has to say."""
        return self._base_url

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat_model(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stream_usage: bool = False,
    ) -> ChatOpenAI:
        """The LangChain model object itself, built once per configuration.

        Public, and the one method on this class that is not on `LLMProvider`:
        `create_react_agent` binds tools onto a `BaseChatModel` and composes it
        into a runnable, so `LangGraphAgent` needs the object rather than a call
        through it. It asks for this shape through its own `ChatModelProvider`
        protocol, which keeps the LangChain type out of the port.
        """
        key = (model, temperature, max_tokens, stream_usage)
        client = self._chat_clients.get(key)
        if client is None:
            client = ChatOpenAI(
                model=model,
                base_url=self._base_url,
                api_key=self._api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                # Asked for explicitly because ChatOpenAI only turns usage
                # reporting on by itself when no `base_url` is set, and this
                # provider always sets one.
                stream_usage=stream_usage,
            )
            self._chat_clients[key] = client
        return client

    def chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        client = self.chat_model(model, temperature=temperature, max_tokens=max_tokens)
        return _text(client.invoke([(m.role, m.content) for m in messages]))

    def ocr(self, model: str, image: bytes, instruction: str) -> str:
        """Transcribe `image` with a vision model — a chat completion, not a route.

        The image goes inline as a base64 data URI in a content block, the same
        OpenAI multimodal shape the other backends take. llama-server accepts it
        only for a server started with a multimodal projector; `/props` reports
        whether this one was, and `model_runtimes.llama_cpp` reads that so the setup
        screen offers this model for OCR only when it can actually read a page.
        """
        encoded = base64.b64encode(image).decode()
        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            {"type": "text", "text": instruction},
        ])
        return _text(self.chat_model(model).invoke([message]))

    # ── Embeddings ────────────────────────────────────────────────────────────

    def _embeddings(self, model: str) -> OpenAIEmbeddings:
        client = self._embed_clients.get(model)
        if client is None:
            client = OpenAIEmbeddings(
                model=model,
                openai_api_base=self._base_url,
                openai_api_key=self._api_key,
                # Off for the same reason as on the other backends, though this
                # server would take the integer arrays: the chunking that comes
                # with it is sized for OpenAI's context, not for whatever
                # `llama-server -c` was given, so it is the wrong cut either way.
                check_embedding_ctx_length=False,
            )
            self._embed_clients[model] = client
        return client

    def embed_query(self, model: str, text: str) -> list[float]:
        return self._embeddings(model).embed_query(text)

    def embed_documents(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        return self._embeddings(model).embed_documents(list(texts))

    def embedding_dimension(self, model: str) -> int:
        """Probe by embedding one token, then remember what came back.

        The probe is also the readiness check the setup screen leans on: this is
        the first request the app makes, so a server that is not up — or one
        started without `--embeddings`, which is how a llama-server is up
        without being able to embed — surfaces here rather than mid-ingest.
        """
        if model not in self._dimensions:
            try:
                self._dimensions[model] = len(self.embed_query(model, PROBE_TEXT))
            except Exception as exc:
                raise BackendUnavailable(LLAMA_CPP_LABEL, self._base_url, str(exc)) from exc
        return self._dimensions[model]

    # ── Reranking ─────────────────────────────────────────────────────────────

    def rerank(self, model: str, query: str, documents: Sequence[str]) -> list[float]:
        """Score every document against `query`, in the order they were given.

        The reason this backend is here. `model` is accepted to satisfy the port
        and then not sent: one llama-server holds one model, chosen when it was
        started, and it echoes back whatever name a request hands it — so a name
        on the wire could only ever disagree with what actually scored.
        """
        documents = list(documents)
        if not documents:
            # The server rejects an empty array with a 400, and a search that
            # fused nothing has no ordering left to improve.
            return []

        url = self._base_url.rstrip("/") + "/rerank"
        try:
            response = requests.post(
                url,
                # No `top_n`: the port promises a score per document, and the
                # caller decides how many survive. Sending one would silently
                # leave every document below the cut at 0.0 — worse than a
                # slower call, because the answer would still look ranked.
                json={"query": query, "documents": documents},
                headers=self._headers,
                timeout=LLAMA_CPP_RERANK_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise BackendUnavailable(LLAMA_CPP_LABEL, url, str(exc)) from exc

        if not response.ok:
            # 501 from a server started without `--rerank`, 400 from one whose
            # model has no rank head to pool over. Both are setup mistakes, and
            # llama.cpp words them well enough to pass straight through.
            raise BackendUnavailable(LLAMA_CPP_LABEL, url, _detail(response))

        results = response.json().get("results")
        if results is None:
            raise BackendUnavailable(LLAMA_CPP_LABEL, url, f"no results in {response.text[:200]}")

        # Scattered back by index rather than read in order: the server sorts
        # its results by score, and this port answers in the order it was asked.
        scores = [0.0] * len(documents)
        for result in results:
            scores[result["index"]] = result["relevance_score"]
        return scores

    @property
    def _headers(self) -> dict[str, str]:
        """The bearer token the OpenAI client sets for itself on the other routes."""
        return {"Authorization": f"Bearer {self._api_key}"}


def _detail(response: requests.Response) -> str:
    """The server's own account of a failure, falling back to the status line."""
    try:
        error = response.json()["error"]
    except Exception:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    return str(error.get("message", error)) if isinstance(error, dict) else str(error)


def _text(reply: BaseMessage) -> str:
    """The reply as a string, whatever shape the model answered in.

    `.content` is typed `str | list`: a model that replies in structured content
    blocks would otherwise hand a list to a caller this provider promised a
    string, and the failure would land wherever that string was next used.
    """
    return reply.content if isinstance(reply.content, str) else str(reply.content)
