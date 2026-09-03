"""Model lifecycle on a running vMLX server — which is mostly to report there is none.

`LmsRuntime` drives LM Studio through a CLI that downloads, loads and unloads on
demand. vMLX offers no equivalent: a server is started with `vmlx serve <model>
--embedding-model <model>` and holds exactly those until it is restarted, and
the only mutable model in the process is the reranker, which loads itself.

So three of the four methods here honestly do nothing, and say why. What remains
is `catalog`, which is the one the setup screen cannot do without — it is how
the dropdowns learn which models this server can actually answer for.
"""

from __future__ import annotations

import os
from typing import Sequence

import requests

from adapters.outbound.model_runtimes.constants import CATALOG_TIMEOUT
from domain.models import ModelCatalog


class VmlxRuntime:
    """Implements `ModelRuntime`."""

    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")

    def catalog(self) -> ModelCatalog:
        """What this server is serving, sorted into what each model can be asked.

        `/v1/models` advertises the chat model, the name it is served under when
        that differs, and the embedding model locked at startup — and types none
        of them, where LM Studio's `/api/v0/models` carries a `type` per entry.
        `/v1/capabilities` names the active chat model and its modalities, so the
        chat models are the ones it points at and the embedder is what is left.

        A server too old to answer `/v1/capabilities` leaves both lists holding
        everything rather than guessing: an id in the wrong dropdown is a choice
        the user can see and avoid, an id in no dropdown is one they cannot make.
        """
        served = self._served()
        if not served:
            return ModelCatalog()

        active = self._active()
        if active is None:
            return ModelCatalog(llm=served, embedding=served)

        loaded, path, modalities = active
        # `path` as well as `loaded`: the two differ when the server was given a
        # `--served-model-name`, and then both names appear in the listing.
        names = {loaded, path, os.path.basename(path)} - {""}
        chat = tuple(model for model in served if model in names)
        return ModelCatalog(
            llm=chat,
            embedding=tuple(model for model in served if model not in names),
            # A vision model answers text prompts like any other, so it belongs
            # in both — see `ModelCatalog`. Only these can read a page image.
            vision=chat if "vision" in modalities else (),
        )

    def download(self, identifier: str) -> str | None:
        """Nothing to fetch ahead of time.

        vMLX resolves a model id against the Hugging Face cache when it loads it
        and pulls what the cache does not hold, so a download is part of the
        first use rather than a step before it. Its Ollama-compatible
        `/api/pull` is an acknowledged no-op, not a way around that.
        """
        return None

    def unload_others(self, keep: Sequence[str]) -> str | None:
        """Nothing to evict.

        One process holds one chat model and one embedder, both bound at
        startup, so there is no residency to reclaim — the memory this would
        free on LM Studio is memory vMLX never took. The reranker is the only
        model that comes and goes, and it unloads its own predecessor when a
        request names a different one.
        """
        return None

    def ensure_loaded(self, identifier: str) -> str | None:
        """Nothing to load, and no route that could.

        `vmlx serve` is the whole loading interface, so a model that is not
        already up cannot be brought up from here. The reranker is the opposite
        case and reaches the same answer: it is absent from `/v1/models` by
        design, right up until the first `/v1/rerank` call loads it.

        Reporting either as a failure would warn on setups that are configured
        correctly. A model that really is missing still surfaces, with a better
        error than this could give: `VMLXProvider.embedding_dimension` probes
        the embedder before the ingest reads a page, and raises
        `BackendUnavailable` naming the model and the address it was wanted at.
        """
        return None

    def _served(self) -> tuple[str, ...]:
        """Ids this server answers for. Empty when it cannot be reached.

        Empty rather than raising, so an unreachable vMLX reads to the setup
        screen as "nothing is up" — the same thing `LmsRuntime` reports, and
        what `ModelCatalog.online` is checked for.
        """
        try:
            response = requests.get(f"{self._base_url}/models", timeout=CATALOG_TIMEOUT)
            response.raise_for_status()
            models = response.json().get("data", [])
        except Exception:
            return ()
        return tuple(model["id"] for model in models if model.get("id"))

    def _active(self) -> tuple[str, str, tuple[str, ...]] | None:
        """The chat model's served name, its path and its modalities.

        None when the route is missing or unreadable, which `catalog` treats as
        "cannot sort these" rather than as a server that is down: `/v1/models`
        already answered, so something is up.
        """
        try:
            response = requests.get(f"{self._base_url}/capabilities", timeout=CATALOG_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None
        return (
            str(payload.get("loaded_model") or ""),
            str(payload.get("model_path") or ""),
            tuple(payload.get("modalities") or ()),
        )
