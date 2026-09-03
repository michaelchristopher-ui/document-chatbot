"""Vendored: the prompt registry's own read client. Do not edit here.

Copied verbatim from `prompt-registry`, whose `src/prompt_registry/client.py`
is written to be vendored exactly this way — single file, stdlib only, no
dependency on the registry package or on anything this app does not already
have, and deliberately kept importable on this app's Python 3.9. Re-copy it to
update; a local change here is a fork nobody upstream knows about, and the next
copy would silently discard it.

Nothing in this app imports it except `registry.py` beside it, which is the
adapter that gives it a port to satisfy, a cache, and a fallback.
"""

# --- BEGIN VENDORED prompt-registry/src/prompt_registry/client.py ------------
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 5.0

#: Matches ``{{ name }}`` placeholders, tolerating internal whitespace.
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class PromptRegistryError(RuntimeError):
    """Raised when the registry answers with an error or cannot be reached."""

    def __init__(
        self, message: str, *, status: int | None = None, code: str = ""
    ) -> None:
        self.status = status
        self.code = code
        super().__init__(message)


class PromptNotFoundError(PromptRegistryError):
    """Raised when the key has no published version, or no such pinned version."""


@dataclass(frozen=True)
class ResolvedPrompt:
    """What a consuming service gets back for one key.

    The key, the text, and which version the text came from. The registry holds
    nothing about how a prompt is meant to be used -- no model, no sampling
    parameters -- so a consumer supplies all of that itself.
    """

    key: str
    version: int
    body: str
    checksum: str
    pinned: bool = False

    def render(self, **values: Any) -> str:
        """Substitute ``{{ placeholder }}`` variables into the template.

        Placeholders with no supplied value are left untouched rather than
        blanked, so a missing variable is visible in the output instead of
        silently producing a subtly wrong prompt.
        """

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            return str(values[name]) if name in values else match.group(0)

        return _PLACEHOLDER.sub(replace, self.body)

    def missing_variables(self, **values: Any) -> list[str]:
        """Placeholders in the body that ``values`` does not cover."""
        found = {match.group(1) for match in _PLACEHOLDER.finditer(self.body)}

        return sorted(found - set(values))


class PromptRegistryClient:
    """A thin read client for the registry."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        api_token: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_token = api_token

    def get(self, key: str, *, version: int | None = None) -> ResolvedPrompt:
        """Fetch one prompt by key.

        Without ``version`` the current published version is returned, so
        consumers pick up new prompts without a redeploy. With it, exactly that
        version is returned.
        """
        path = f"/v1/resolve/{urllib.parse.quote(key, safe='')}"
        if version is not None:
            path = f"{path}?{urllib.parse.urlencode({'version': version})}"

        payload = self._request(path)
        data = payload.get("data", {})

        return ResolvedPrompt(
            key=data.get("slug", key),
            version=int(data["version"]),
            body=data["body"],
            checksum=data.get("checksum", ""),
            pinned=bool(data.get("pinned", version is not None)),
        )

    def _request(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(f"{self._base_url}{path}", method="GET")
        request.add_header("Accept", "application/json")
        if self._api_token:
            request.add_header("Authorization", f"Bearer {self._api_token}")

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))  # type: ignore[no-any-return]
        except urllib.error.HTTPError as error:
            raise _from_http_error(error) from error
        except urllib.error.URLError as error:
            raise PromptRegistryError(
                f"could not reach the prompt registry: {error.reason}"
            ) from error


def _from_http_error(error: urllib.error.HTTPError) -> PromptRegistryError:
    """Turn the registry's error envelope into a typed exception."""
    code, message = "", error.reason or "request failed"

    try:
        payload = json.loads(error.read().decode("utf-8"))
        body = payload.get("error", {})
        code, message = body.get("code", ""), body.get("message", message)
    except (ValueError, OSError):
        pass

    if error.code == 404:
        return PromptNotFoundError(message, status=error.code, code=code)

    return PromptRegistryError(message, status=error.code, code=code)
# --- END VENDORED -----------------------------------------------------------
