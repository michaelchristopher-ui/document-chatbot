"""Constants for `adapters.inbound`."""

from __future__ import annotations

from domain.constants import HIGH, LOW, MEDIUM

# ── Setup and chat screen — `streamlit_ui` ────────────────────────────────────

STRATEGIES = ["fixed", "recursive", "semantic"]
# What the setup screen decides, and what `_load` is keyed on.
CONFIG_KEYS = (
    "backend_url", "chat_model", "ocr_model", "embed_model",
    "reranker_model", "judge_model", "chunking_strategy",
)
SESSION_KEYS = ("ready", *CONFIG_KEYS, "messages", "thread_id")

# A tooltip is a native `title`, so its width is the browser's to decide and its
# line breaks are ours — hence the hard wrap.
TOOLTIP_CHARS = 360
TOOLTIP_WIDTH = 68
CITATION_STYLE = """
<style>
.citation {
    color: #ff4b4b;
    cursor: help;
    font-size: 0.75em;
    font-weight: 600;
    vertical-align: super;
    text-decoration: underline dotted;
    text-underline-offset: 0.2em;
}
</style>
"""

# What each band looks like at a glance. A dot rather than a coloured number:
# the number is the reading, and a reader scanning a thread wants to know which
# answers to go back and check before reading any of them.
CONFIDENCE_DOTS = {HIGH: "🟢", MEDIUM: "🟡", LOW: "🔴"}
UNSCORED_DOT = "⚪"


# ── Statistics page — `streamlit_statistics` ──────────────────────────────────

# How many turns to read. Anything here is loaded into memory and aggregated in
# Python, which is why it is a choice and not "everything".
WINDOW_OPTIONS = (100, 500, 2000)
DEFAULT_WINDOW = 500
RECENT_SEARCHES = 100
RECENT_TURNS = 20
QUESTION_LABEL_CHARS = 90
# A judged answer costs a model call, so a batch is capped at something a person
# will actually wait through rather than the whole backlog at once.
MAX_BATCH = 50
DEFAULT_BATCH = 5
