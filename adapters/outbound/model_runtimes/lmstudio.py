"""Model lifecycle via the `lms` CLI and LM Studio's management API."""

from __future__ import annotations

import subprocess
import time
from typing import Sequence

import requests

from adapters.outbound.model_runtimes.constants import (
    CATALOG_TIMEOUT,
    DEFAULT_LMS_BIN,
    DOWNLOAD_TIMEOUT,
    EMBEDDING_TYPES,
    LLM_TYPES,
    MAX_WAIT,
    POLL_INTERVAL,
    UNLOAD_TIMEOUT,
    VISION_TYPES,
)
from domain.models import ModelCatalog


class LmsRuntime:
    """Implements `ModelRuntime`."""

    def __init__(
        self,
        base_url: str,
        lms_bin: str = DEFAULT_LMS_BIN,
        poll_interval: int = POLL_INTERVAL,
        max_wait: int = MAX_WAIT,
    ):
        self._base_url = base_url
        self._lms_bin = lms_bin
        self._poll_interval = poll_interval
        self._max_wait = max_wait

    @property
    def _management_url(self) -> str:
        """The management API lives one level above the OpenAI-compatible /v1 root."""
        return self._base_url.rstrip("/").removesuffix("/v1")

    def catalog(self) -> ModelCatalog:
        try:
            response = requests.get(f"{self._management_url}/api/v0/models", timeout=CATALOG_TIMEOUT)
            response.raise_for_status()
            models = response.json().get("data", [])
        except Exception:
            return ModelCatalog()
        return ModelCatalog(
            llm=tuple(m["id"] for m in models if m.get("type") in LLM_TYPES),
            embedding=tuple(m["id"] for m in models if m.get("type") in EMBEDDING_TYPES),
            vision=tuple(m["id"] for m in models if m.get("type") in VISION_TYPES),
        )

    def download(self, identifier: str) -> str | None:
        result = subprocess.run(
            [self._lms_bin, "get", identifier, "-y"],
            capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT,
        )
        if result.returncode != 0:
            return (result.stderr or result.stdout).strip()
        return None

    def unload_others(self, keep: Sequence[str]) -> str | None:
        """Evict every loaded model except `keep`, freeing memory for the load ahead.

        LM Studio never reuses a resident model: each `lms load` of an id that is
        already up stacks a *second* instance under a `:2` suffix, at full memory
        cost. On the 12 GB machine `config` sizes the recommendations for, a
        couple of stale instances are enough to leave no room for the KV cache.

        Ids are compared exactly, which is what makes those duplicates go: `…:2`
        is not `keep`'s id, so it is an "other" and gets unloaded, while the
        already-resident original stays and costs no reload.
        """
        wanted = set(keep)
        failures = []
        for identifier in self._loaded():
            if identifier in wanted:
                continue
            try:
                result = subprocess.run(
                    [self._lms_bin, "unload", identifier],
                    capture_output=True, text=True, timeout=UNLOAD_TIMEOUT,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                # Reported, never raised: callers treat freeing memory as
                # best-effort, so a missing `lms` binary or a hung unload must
                # not take down a load that would have succeeded anyway.
                failures.append(f"`{identifier}`: {exc}")
                continue
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                failures.append(
                    f"`{identifier}`: {detail or f'`lms unload` exited with code {result.returncode}'}"
                )
        return "; ".join(failures) or None

    def ensure_loaded(self, identifier: str) -> str | None:
        """Start `lms load` and poll the API until the model reports as loaded.

        Returns early when the model is already up, which is what makes the
        "ensure" honest: `lms load` does not reuse a resident model, it stacks a
        fresh instance under a `:2` suffix at full memory cost. Without this,
        every visit to the setup screen would pile another copy onto a machine
        `config` already sizes with no slack.

        The model is loaded *under the identifier asked for*, so what this polls
        for is the same string the rest of the app names the model by — see the
        flags below for why both of them are load-bearing.
        """
        if self._is_loaded(identifier):
            return None

        process = subprocess.Popen(
            [
                self._lms_bin,
                "load",
                identifier,
                # Non-interactive, and not optional. A model key that matches
                # more than one installed quantisation — which
                # `text-embedding-nomic-embed-text-v1.5` is the moment both a
                # q8_0 and a q4_k_m are present, and it is this backend's own
                # recommended embedder — makes `lms load` print a picker and
                # wait for a keystroke. With stdout piped and nobody at a
                # keyboard that blocks until `_max_wait`, so a 140 MB embedder
                # takes thirty minutes and then reports a timeout. `-y` takes
                # the first match instead.
                "-y",
                # Name the loaded instance after what we asked for. Without
                # this, LM Studio registers the concrete build it chose
                # (`…-v1.5@q8_0`), `_is_loaded` compares exact strings and
                # never matches, and the poll below runs to the deadline even
                # though the model came up seconds in.
                "--identifier",
                identifier,
            ],
            # An interactive prompt this flag does not cover should fail rather
            # than wait: reading EOF makes `lms` exit and surface its message
            # through the returncode path below.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        deadline = time.time() + self._max_wait
        while time.time() < deadline:
            time.sleep(self._poll_interval)
            if self._is_loaded(identifier):
                process.wait(timeout=10)
                return None

            if process.poll() is not None:
                _, stderr = process.communicate()
                if process.returncode != 0:
                    return stderr.strip() or f"`lms load {identifier}` exited with code {process.returncode}"
                return None

        process.kill()
        return f"Timed out after {self._max_wait // 60} minutes waiting for `{identifier}` to load."

    def _is_loaded(self, identifier: str) -> bool:
        return identifier in self._loaded()

    def _loaded(self) -> tuple[str, ...]:
        """Ids currently resident, including the `:2`-style duplicate instances.

        Empty when the server cannot be reached, so callers read an unreachable
        LM Studio as "nothing is up" rather than failing.
        """
        try:
            response = requests.get(f"{self._management_url}/api/v0/models", timeout=CATALOG_TIMEOUT)
            response.raise_for_status()
            models = response.json().get("data", [])
        except Exception:
            return ()
        return tuple(m["id"] for m in models if m.get("state") == "loaded")
