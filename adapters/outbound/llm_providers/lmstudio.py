"""LM Studio behind `ports.outbound.LLMProvider`.

The one module that knows this backend exists. Its address, its placeholder
credential, the OpenAI-compatible routes it serves and the three places it
departs from that schema are all spelled here; every adapter above holds an
`LLMProvider` and names no backend at all.

LM Studio speaks the OpenAI chat and embeddings API, so the OpenAI client is
what talks to it — pointed at a local base URL rather than api.openai.com. The
departures are worth knowing, because they are why this is not four lines:

- `/v1/embeddings` rejects the pre-tokenised integer arrays the OpenAI client
  sends by default, so that behaviour is turned off.
- Usage reporting has to be asked for, because the client only enables it by
  itself against the default base URL — and this backend never is one.
- `/rerank` is not an OpenAI route at all, so it is posted by hand.
"""

from __future__ import annotations

import base64
from typing import Sequence

import requests
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from adapters.outbound.llm_providers.constants import (
    LM_STUDIO_API_KEY,
    LM_STUDIO_LABEL,
    LM_STUDIO_RERANK_TIMEOUT,
    PROBE_TEXT,
)
from domain.errors import BackendUnavailable
from domain.models import ChatMessage


class LMStudioProvider:
    """Implements `LLMProvider`.

    Clients are built on first use and kept, keyed by the configuration they
    were built with. Every one of them opens an HTTP connection pool, and a
    judge that made a fresh client per answer would open one per answer.
    """

    def __init__(self, base_url: str, api_key: str = LM_STUDIO_API_KEY):
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
                # provider always sets one. It is what puts `stream_options:
                # {"include_usage": true}` on the request; a server that ignores
                # that simply sends no usage, and the turn records none.
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
        the OpenAI multimodal shape LM Studio accepts. So the model named here
        has to be vision-capable; see `config` on why it is the chat model.
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
                # which LM Studio rejects: "'input' field must be a string or an
                # array of strings".
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
        the first request the app makes, so an unloaded model or a server that
        is not up surfaces here as `BackendUnavailable` rather than mid-ingest.
        """
        if model not in self._dimensions:
            try:
                self._dimensions[model] = len(self.embed_query(model, PROBE_TEXT))
            except Exception as exc:
                raise BackendUnavailable(LM_STUDIO_LABEL, self._base_url, str(exc)) from exc
        return self._dimensions[model]

    # ── Reranking ─────────────────────────────────────────────────────────────

    def rerank(self, model: str, query: str, documents: Sequence[str]) -> list[float]:
        url = self._base_url.rstrip("/") + "/rerank"
        response = requests.post(
            url,
            json={"model": model, "query": query, "documents": list(documents)},
            timeout=LM_STUDIO_RERANK_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if "results" not in payload:
            # LM Studio answers 200 with {"error": ...} for routes it does not
            # serve, and its REST server has no /rerank route at all.
            raise BackendUnavailable(LM_STUDIO_LABEL, url, str(payload))

        scores = [0.0] * len(documents)
        for result in payload["results"]:
            scores[result["index"]] = result["relevance_score"]
        return scores


def _text(reply: BaseMessage) -> str:
    """The reply as a string, whatever shape the model answered in.

    `.content` is typed `str | list`: a model that replies in structured content
    blocks would otherwise hand a list to a caller this provider promised a
    string, and the failure would land wherever that string was next used.
    """
    return reply.content if isinstance(reply.content, str) else str(reply.content)
