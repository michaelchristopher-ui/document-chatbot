"""The prompts as this app ships them, behind the same port as a registry."""

from __future__ import annotations


class BuiltinPrompts:
    """Implements `PromptLibrary` by always answering with the default.

    A null object rather than a `None` the callers check for. The three adapters
    that send a prompt to a model then hold a `PromptLibrary` unconditionally,
    and none of them carries a branch for the ordinary case of no registry —
    which is the case this app is normally run in.

    Holding no state and doing no work, one instance would do. It is still
    constructed per build, because a `Config` is what decides which
    implementation is wired and that is `composition`'s decision to make once.
    """

    def text(self, key: str, default: str) -> str:
        return default
