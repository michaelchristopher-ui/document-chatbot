"""Model lifecycle on a running llama-server — which is mostly to report there is none.

`LmsRuntime` drives LM Studio through a CLI that downloads, loads and unloads on
demand. llama-server offers no equivalent and never will: the model is chosen on
the command line, loaded before the port is bound, and held until the process is
restarted. There is not even a lazily-loaded reranker the way vMLX has one — the
reranker *is* the model, on a server started with `--rerank`.

So three of the four methods here honestly do nothing, and say why. What remains
is `catalog`, which is the one the setup screen cannot do without — it is how
the dropdowns learn which model this server can answer for. Singular: one
process, one model.
"""

from __future__ import annotations

from typing import Sequence

import requests

from adapters.outbound.model_runtimes.constants import CATALOG_TIMEOUT
from domain.models import ModelCatalog


class LlamaCppRuntime:
    """Implements `ModelRuntime`."""

    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        # `/props` is mounted at the server root, not under the OpenAI-compatible
        # prefix the rest of this talks to — `/v1/props` is a 404.
        self._root = self._base_url.removesuffix("/v1")

    def catalog(self) -> ModelCatalog:
        """The one model this server holds, offered for everything it might be.

        `/v1/models` lists it and types it no further — where LM Studio's
        `/api/v0/models` carries a `type` per entry, this says only that a model
        is up. What the model can actually do was decided by the flags it was
        started with (`--embeddings`, `--rerank`), and none of them are
        reported, so the id goes in both lists rather than being guessed into
        one: an id in the wrong dropdown is a choice the user can see and avoid,
        an id in no dropdown is one they cannot make.

        `/props` is the exception, because it does answer one of the questions:
        `modalities.vision` says whether a multimodal projector was loaded, and
        a server without one cannot read a page image at all.
        """
        served = self._served()
        if not served:
            return ModelCatalog()
        return ModelCatalog(
            llm=served,
            embedding=served,
            # A vision model answers text prompts like any other, so it belongs
            # in both — see `ModelCatalog`. Only these can read a page image.
            vision=served if self._vision() else (),
        )

    def download(self, identifier: str) -> str | None:
        """Nothing to fetch ahead of time.

        `llama-server -hf <repo>` resolves an id against the Hugging Face cache
        as it starts and pulls what the cache does not hold, so a download
        belongs to launching the server rather than to a step before it — and by
        the time this app can reach the server, it has already happened.
        """
        return None

    def unload_others(self, keep: Sequence[str]) -> str | None:
        """Nothing to evict.

        One process holds one model, bound at startup, so there is no residency
        to reclaim: the memory this frees on LM Studio is memory llama-server
        never took. A second model means a second process on a second port,
        which this app addresses as a second backend rather than as a load here.
        """
        return None

    def ensure_loaded(self, identifier: str) -> str | None:
        """Nothing to load, and no route that could.

        The command line is the whole loading interface, so a model that is not
        already up cannot be brought up from here. Reporting that as a failure
        would warn on setups that are configured correctly — and a model that
        really is missing still surfaces, with a better error than this could
        give: `LlamaCppProvider.embedding_dimension` probes the embedder before
        the ingest reads a page, and `rerank` passes through the server's own
        501 for the one mistake that looks like a working setup — a reranker
        loaded on a server started without `--rerank`.
        """
        return None

    def _served(self) -> tuple[str, ...]:
        """Ids this server answers for. Empty when it cannot be reached.

        Empty rather than raising, so an unreachable llama-server reads to the
        setup screen as "nothing is up" — the same thing the other runtimes
        report, and what `ModelCatalog.online` is checked for.
        """
        try:
            response = requests.get(f"{self._base_url}/models", timeout=CATALOG_TIMEOUT)
            response.raise_for_status()
            models = response.json().get("data", [])
        except Exception:
            return ()
        return tuple(model["id"] for model in models if model.get("id"))

    def _vision(self) -> bool:
        """Whether a multimodal projector is loaded. False when `/props` is unreadable.

        False rather than unknown: `catalog` uses this to decide whether to
        offer the model for OCR, and offering one that cannot see would fail
        silently — a page of invented text at a time.
        """
        try:
            response = requests.get(f"{self._root}/props", timeout=CATALOG_TIMEOUT)
            response.raise_for_status()
            modalities = response.json().get("modalities") or {}
        except Exception:
            return False
        return bool(modalities.get("vision"))
