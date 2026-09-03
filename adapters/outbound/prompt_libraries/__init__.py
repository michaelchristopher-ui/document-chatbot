"""Where the prompts come from, behind `ports.outbound.PromptLibrary`.

Two implementations and no registry of backends, unlike the vector stores: the
choice here is only whether a prompt store is configured at all.

- `BuiltinPrompts` answers every call with the default it was handed, which is
  the constant in `domain.constants`. What an app with no store configured
  runs on, and the default.
- `PromptRegistryLibrary` reads a published prompt from the `prompt-registry`
  service, falling back to the same constant whenever it cannot.

Both are read-only. Nothing in this app writes a prompt — editing one is done
in the registry's own console, and this side only ever asks what is live.
"""

from adapters.outbound.prompt_libraries.local import BuiltinPrompts
from adapters.outbound.prompt_libraries.registry import PromptRegistryLibrary

__all__ = ["BuiltinPrompts", "PromptRegistryLibrary"]
