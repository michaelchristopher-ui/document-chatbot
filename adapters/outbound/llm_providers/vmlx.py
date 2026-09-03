"""vMLX behind `ports.outbound.LLMProvider`.

The one module that knows this backend exists. Its address, its optional
credential, the OpenAI-compatible routes it serves and the places it departs
from that schema are all spelled here; every adapter above holds an
`LLMProvider` and names no backend at all.

vMLX (github.com/jjang-ai/vmlx) is an MLX inference server for Apple Silicon.
It speaks the same OpenAI chat and embeddings API LM Studio does, so the same
client talks to both. What earns it an adapter of its own is `/v1/rerank`:
LM Studio's REST server has no such route, so a reranker is the one thing this
app asks for that its default backend cannot answer at all — see
`application.retrieval.HybridRetriever` for what is lost without one.

Where it departs from the OpenAI schema, and from LM Studio:

- `/v1/rerank` is Cohere/Jina-shaped rather than OpenAI, so it is posted by
  hand. `top_n` is deliberately not sent: omitted, the server scores every
  document, and this port promises a score per document rather than a shortlist.
- Failures arrive as a 4xx/5xx with a FastAPI `{"detail": ...}` body, where
  LM Studio answers 200 with `{"error": ...}`. So the status is what is checked
  here, and the body only decides what the error says.
- `/v1/embeddings` types `input` as `str | list[str]`, so the OpenAI client's
  pre-tokenised integer arrays are rejected — the same setting as LM Studio
  needs, for a validation error rather than a parse one.
- Authentication is off until the server is started with `--api-key`, and is a
  bearer token when it is on. `VMLX_API_KEY` is where this reads one.

One process serves the chat model it was started with, the `--embedding-model`
it was given, and a reranker it loads lazily on the first `/v1/rerank` call —
swapping that reranker whenever a request names a different one. So a single
provider can answer all four capabilities; `adapters.outbound.model_runtimes.vmlx`
covers what that costs the setup screen.
"""

from __future__ import annotations

import base64
from typing import Sequence

import requests
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from adapters.outbound.llm_providers.constants import (
    PROBE_TEXT,
    VMLX_API_KEY,
    VMLX_LABEL,
    VMLX_RERANK_TIMEOUT,
)
from domain.errors import BackendUnavailable
from domain.models import ChatMessage


class VMLXProvider:
    """Implements `LLMProvider`.

    Clients are built on first use and kept, keyed by the configuration they
    were built with. Every one of them opens an HTTP connection pool, and a
    judge that made a fresh client per answer would open one per answer.
    """

    def __init__(self, base_url: str, api_key: str = VMLX_API_KEY):
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

        The image goes inline as a base64 data URI in a content block, which is
        the OpenAI multimodal shape vMLX accepts for the VLMs it serves. So the
        model named here has to be one of those: a text-only model takes the
        request and answers about an image it never received.
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
                # Otherwise the client pre-tokenises and posts integer arrays,
                # which vMLX's `EmbeddingRequest` types as `str | list[str]` and
                # rejects with a 422 before the model is ever reached.
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
        started without `--embedding-model`, which is the vMLX way to be up
        without being able to embed — surfaces here rather than mid-ingest.
        """
        if model not in self._dimensions:
            try:
                self._dimensions[model] = len(self.embed_query(model, PROBE_TEXT))
            except Exception as exc:
                raise BackendUnavailable(VMLX_LABEL, self._base_url, str(exc)) from exc
        return self._dimensions[model]

    # ── Reranking ─────────────────────────────────────────────────────────────

    def rerank(self, model: str, query: str, documents: Sequence[str]) -> list[float]:
        """Score every document against `query`, in the order they were given.

        The reason this backend is here. vMLX loads the named cross-encoder on
        the first call and holds it until a request names a different one, so
        the model id travels with each request rather than being configured
        once — which is also what lets one server answer for chat, embeddings
        and reranking at the same address.
        """
        url = self._base_url.rstrip("/") + "/rerank"
        try:
            response = requests.post(
                url,
                # No `top_n`: the port promises a score per document, and the
                # caller decides how many survive. Sending one would silently
                # leave every document below the cut at 0.0 — worse than a
                # slower call, because the answer would still look ranked.
                json={"model": model, "query": query, "documents": list(documents)},
                headers=self._headers,
                timeout=VMLX_RERANK_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise BackendUnavailable(VMLX_LABEL, url, str(exc)) from exc

        if not response.ok:
            # FastAPI's shape: 400 for a query, a document list or a model id it
            # will not take, 404 from a vMLX too old to serve this route at all.
            raise BackendUnavailable(VMLX_LABEL, url, _detail(response))

        results = response.json().get("results")
        if results is None:
            raise BackendUnavailable(VMLX_LABEL, url, f"no results in {response.text[:200]}")

        # Scattered back by index rather than read in order: the server sorts
        # its results by score, and this port answers in the order it was asked.
        scores = [0.0] * len(documents)
        for result in results:
            scores[result["index"]] = result["relevance_score"]
        return scores

    @property
    def _headers(self) -> dict[str, str]:
        """A bearer token, which an unguarded server accepts and ignores."""
        return {"Authorization": f"Bearer {self._api_key}"}


def _detail(response: requests.Response) -> str:
    """What the server said went wrong, or the status if it did not say.

    vMLX raises `HTTPException`, so the reason is in `detail` — but a proxy in
    front of it, or a crash inside it, answers with something else entirely, and
    an adapter that assumed the shape would report a `KeyError` for a 502.
    """
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return f"HTTP {response.status_code}: {response.text[:200]}"


def _text(reply: BaseMessage) -> str:
    """The reply as a string, whatever shape the model answered in.

    `.content` is typed `str | list`: a model that replies in structured content
    blocks would otherwise hand a list to a caller this provider promised a
    string, and the failure would land wherever that string was next used.
    """
    return reply.content if isinstance(reply.content, str) else str(reply.content)
