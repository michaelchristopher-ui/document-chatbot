"""Implements `PromptLibrary` against the `prompt-registry` service.

What the registry is, from here: a key/value store of prompt bodies, each with a
version number and a pointer to the version in use. It holds the text and
nothing else — no model, no temperature, no notion of what the text is for —
which is why this adapter reads a body and stops. Which model the body is sent
to, at what temperature, with what tokens, is decided in `config` and
`composition` exactly as it was before a registry existed.

Two things make it safe to put on the answer path.

*A failure is the built-in prompt.* Any exception — the server down, a key never
published, a body that is not a string — resolves to the `default` the caller
passed, which is the constant in `domain.constants`. So the worst a broken
registry costs is that the app answers the way it does with no registry at all.
Nothing here raises, and nothing here logs a stack trace per turn: the outcome
is recorded once per key per window, in `report`, so the UI can say which
prompts a run is actually using.

*A hit is a dict lookup.* Bodies are cached for `DEFAULT_TTL_SECONDS`, so the
number of round trips is bounded by the number of keys and not by the number of
questions.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, NamedTuple, Optional, Tuple

from adapters.outbound.prompt_libraries.client import (
    PromptRegistryClient,
    PromptRegistryError,
    ResolvedPrompt,
)
from adapters.outbound.prompt_libraries.constants import (
    DEFAULT_TTL_SECONDS,
    FAILURE_TTL_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)
from domain.constants import (
    PARTIAL_REFUSAL_SENTENCE,
    PLACEHOLDER_PARTIAL_REFUSAL,
    PLACEHOLDER_REFUSAL,
    REFUSAL_SENTENCE,
)


class Resolution(NamedTuple):
    """Where one key's text came from, for a reader who wants to know.

    Worth reporting because the two outcomes are indistinguishable in the
    answer: a prompt served from the registry and the built-in one that stood in
    for it are both just a string by the time a model sees them. Which of them a
    run is on is the first thing to check when an answer changed and nothing in
    this repository did.
    """

    key: str
    #: None when the built-in prompt was used, which is what `registered` reads.
    version: Optional[int]
    #: Empty unless the registry declined or could not be reached.
    error: str

    @property
    def registered(self) -> bool:
        return self.version is not None


class _Entry(NamedTuple):
    """One cached lookup and when it stops being trusted."""

    body: Optional[str]
    expires_at: float
    version: Optional[int]
    error: str


class PromptRegistryLibrary:
    """Implements `PromptLibrary`."""

    def __init__(
        self,
        base_url: str,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        api_token: Optional[str] = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        clock: Optional[object] = None,
    ):
        self._client = PromptRegistryClient(
            base_url, timeout=timeout, api_token=api_token
        )
        self._ttl = ttl_seconds
        self._cache: Dict[str, _Entry] = {}
        # Streamlit answers two questions in two threads against one cached
        # `Application`, so this cache has concurrent readers by construction.
        # The lock covers the dict only, never the HTTP call: holding it across a
        # request would make one slow lookup block every other key.
        self._lock = threading.Lock()
        self._now = clock if callable(clock) else _monotonic

    def text(self, key: str, default: str) -> str:
        """The published body for `key`, or `default`. Never raises."""
        cached = self._cached(key)
        if cached is not None:
            return self._fill(cached, default)

        body, version, error = self._fetch(key)
        expires = self._now() + (self._ttl if error == "" else FAILURE_TTL_SECONDS)

        with self._lock:
            self._cache[key] = _Entry(body, expires, version, error)

        return self._fill(body, default)

    def report(self) -> Tuple[Resolution, ...]:
        """What every key looked up so far resolved to, newest state per key.

        A snapshot of the cache rather than a running log: the same key asked
        twenty times in a session is one line here, which is what a reader
        wants. Keys never asked for are absent — this reports what happened,
        not what could.
        """
        with self._lock:
            entries = sorted(self._cache.items())

        return tuple(
            Resolution(key=key, version=entry.version, error=entry.error)
            for key, entry in entries
        )

    def _cached(self, key: str) -> Optional[str]:
        """The cached body for `key`, or None when there is none worth using.

        None covers both "never looked up" and "looked up and expired", and also
        a lookup that failed — a cached failure means "do not ask again yet",
        and the caller's default stands in until it expires.
        """
        with self._lock:
            entry = self._cache.get(key)

        if entry is None or entry.expires_at <= self._now():
            return None

        return entry.body

    def _fetch(self, key: str) -> Tuple[Optional[str], Optional[int], str]:
        """One resolve call, with every failure turned into a reason.

        Deliberately catches `Exception` and not just `PromptRegistryError`. The
        client raises that for the failures it anticipated — an unreachable
        server, an error envelope — but this sits on the answer path, and a
        malformed payload that came back as a `KeyError` or a `TypeError` must
        cost the built-in prompt rather than the turn.
        """
        try:
            resolved = self._client.get(key)
        except PromptRegistryError as error:
            return None, None, str(error)
        except Exception as error:  # noqa: BLE001 - see the docstring
            return None, None, f"{type(error).__name__}: {error}"

        if not isinstance(resolved.body, str) or not resolved.body.strip():
            # A published prompt the registry cannot have meant: the API rejects
            # a blank body, so this is a shape nothing should produce. Treated as
            # a failure rather than sent to a model.
            return None, None, f"{key} resolved to an empty body"

        return resolved.body, resolved.version, ""

    def _fill(self, body: Optional[str], default: str) -> str:
        """Substitute the sentences a registered prompt writes as placeholders.

        Only for a registered body. The built-in prompts are f-strings that
        already carry these sentences, and running the substitution over one
        would be a no-op that implied otherwise.

        Unmatched placeholders are left in place by the client's `render`, which
        is what makes a typo in a prompt visible in the answer instead of
        silently blank — see `domain.constants.PLACEHOLDER_REFUSAL` for why
        these two in particular have to survive the round trip.
        """
        if body is None:
            return default

        # Rendered through the client's own `ResolvedPrompt` rather than a regex
        # here: the placeholder syntax is the registry's, and a second
        # definition of it in this file would be free to drift from it.
        return ResolvedPrompt(key="", version=0, body=body, checksum="").render(
            **{
                PLACEHOLDER_REFUSAL: REFUSAL_SENTENCE,
                PLACEHOLDER_PARTIAL_REFUSAL: PARTIAL_REFUSAL_SENTENCE,
            }
        )


def _monotonic() -> float:
    """A clock that cannot go backwards, which a TTL depends on.

    `time.monotonic` rather than `time.time`: a wall clock adjusted backwards —
    NTP, a laptop waking up — would leave every cached prompt looking fresh for
    however long the correction was.
    """
    return time.monotonic()
