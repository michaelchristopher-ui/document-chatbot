"""Every constant the domain reads, in one place.

Grouped by the module that reads each set. The order is alphabetical except
where one group is built from another — `domain.statistics` comes before
`domain.confidence` because the confidence band is scaled across
`WEAK_RETRIEVAL`, which is also what a single file is worth having for: that is
one line drawn between a match worth keeping and one that is not, both modules
read it, and they have to draw it in the same place or the app comes to
disagree with itself about what "weak" means.

Type aliases stay with the modules that define them — a `Callable` shape is
part of a module's interface, not a value it was configured with.
"""

from __future__ import annotations

import re


# ── Chunking — `domain.chunking` ──────────────────────────────────────────────

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_TEXT_LEN = 100

SEMANTIC_SIMILARITY_THRESHOLD = 0.75
SEMANTIC_WINDOW_SIZE = 2

_HEADER_RE = re.compile(
    r'(?:^|\n)(?:[A-Z][A-Z\s\-]{3,}[A-Z]|Chapter\s+\d+|CHAPTER\s+\d+|\d+(?:\.\d+)*\s+[A-Z])(?=\s|$)',
    re.MULTILINE,
)
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_ABBREV_MAP = {
    "Mr.": "Mr\x00", "Mrs.": "Mrs\x00", "Dr.": "Dr\x00",
    "Fig.": "Fig\x00", "No.": "No\x00", "vs.": "vs\x00",
    "e.g.": "eg\x00", "i.e.": "ie\x00", "et al.": "etal\x00",
}
_ABBREV_RESTORE = {v: k for k, v in _ABBREV_MAP.items()}


# ── Citations — `domain.citations` ────────────────────────────────────────────

NO_RESULTS_MESSAGE = "No relevant content found in the loaded documents."

FULL_REFUSAL = "full"
PARTIAL_REFUSAL = "partial"

# What the system prompt asks the model to write: `[1]`, or `[1, 3]` for a claim
# drawing on more than one block.
_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Fenced blocks and inline spans, so a citation-shaped literal inside code stays
# exactly as it was written.
_CODE_RE = re.compile(r"```.*?```|``.*?``|`[^`\n]*`", re.DOTALL)


# ── Statistics — `domain.statistics` ──────────────────────────────────────────

MEDIAN = 0.5
P95 = 0.95

# Below this cosine similarity, nothing in the corpus really matched the question.
# An answer built on such passages is the one most likely to have been invented,
# and this is the only warning available *before* the model writes anything.
WEAK_RETRIEVAL = 0.5

# Below this gap between the best match and the rest of its own ranking, nothing
# stood out: the embedder returned a flat field where one passage should have won,
# and the order the model read is close to arbitrary.
#
# Calibrated against this corpus, not a universal figure. Measured 2026-08-25 with
# nomic-embed-text-v1.5 over 2,028 chunks: searches that retrieved well separated by
# 0.020, 0.037, 0.061 and 0.084, and the two that failed by 0.0018 and 0.0051 — the
# second of those ranking a passage from the wrong document above the right one.
# 0.01 sits in the gap with margin on both sides. Six observations is not a
# calibration, so treat a flag as a prompt to look rather than a verdict; the
# labelled eval set is what would settle it.
FLAT_RETRIEVAL = 0.01

GROUNDED = "grounded"
REFUSED = "refused"
UNATTRIBUTED = "unattributed"

TURN_AXIS = "turn"
TOTAL_SERIES = "total ms"
FIRST_TOKEN_SERIES = "to first token ms"


# ── Deduplication — `domain.dedup` ────────────────────────────────────────────

DEDUP_COSINE_THRESHOLD = 0.95
DEDUP_MINHASH_THRESHOLD = 0.85
DEDUP_MINHASH_NUM_PERM = 128


# ── Fusion — `domain.fusion` ──────────────────────────────────────────────────

RRF_K = 60


# ── Interactions — `domain.interactions` ──────────────────────────────────────

ARM_BOTH = "both"
ARM_DENSE = "dense"
ARM_KEYWORD = "keyword"
# Not "neither", which is impossible for a passage that was returned: it means the
# search that returned it did not record where it came from. Every row written
# before arm attribution existed reads as this, and folding them into a real arm
# would turn silence into a finding.
ARM_UNATTRIBUTED = "unattributed"

# Below this an answer is saying things its sources do not support. A judgement
# is a model's opinion, so this is a line for sorting attention, not a verdict.
FAITHFUL_THRESHOLD = 0.7


# ── Prompts — `domain.prompts` ────────────────────────────────────────────────

# The two sentences rules 7 and 8 instruct the model to write verbatim, spliced in
# below rather than written out twice. `domain.citations.refusal_kind` recognises a
# refusal by matching them, so a second copy free to drift would stop recognising one
# the moment the wording here was reworded — and it would fail silently, reading every
# refusal as an ordinary answer that happened to cite nothing.
REFUSAL_SENTENCE = "The provided documents do not contain information about"
PARTIAL_REFUSAL_SENTENCE = "The documents do not contain information about"

# What each prompt below is addressed by when it is read from a `PromptLibrary`
# rather than from this file. Namespaced, because a registry is shared: these
# are this app's prompts among whoever else's.
#
# The prompts here stay the defaults, and they are not dead code — an app with
# no prompt library configured runs entirely on them, and one with a library
# falls back to them for any key it has not been given. So a key added here
# needs its default here too.
PROMPT_KEY_SYSTEM = "chatbot.system"
PROMPT_KEY_SEARCH_TOOL = "chatbot.search-tool"
PROMPT_KEY_OCR = "chatbot.ocr-instruction"
PROMPT_KEY_JUDGE = "chatbot.judge"
PROMPT_KEY_JUDGE_REQUEST = "chatbot.judge-request"

# The two sentences a registered system prompt splices them into. `SYSTEM_PROMPT`
# below is an f-string and gets them at import; a body that came from a registry
# cannot be, so it writes `{{refusal_sentence}}` and the library substitutes.
#
# This is not cosmetic. `domain.citations.refusal_kind` recognises a refusal by
# matching these sentences in the answer, so a registered prompt that spelled
# them out in its own words — or reworded them slightly — would still produce
# refusals, and this app would read every one of them as an ordinary answer that
# happened to cite nothing. Writing the placeholder is what keeps the two ends
# tied together through the registry.
PLACEHOLDER_REFUSAL = "refusal_sentence"
PLACEHOLDER_PARTIAL_REFUSAL = "partial_refusal_sentence"

SYSTEM_PROMPT = f"""You are a document assistant. Answer questions exclusively from the numbered context blocks returned by the search_documents tool. You have no authority to use knowledge outside these blocks.

CONTEXT FORMAT
search_documents returns results as numbered blocks. Each block carries the page number and the title of the document it came from. The block below shows the shape of that output and nothing more — it is a layout illustration, it is not context, it did not come from any loaded document, and you must never answer from it or mention what it contains:

  --- BEGIN LAYOUT ILLUSTRATION (not document content) ---
  [1] Page 3 | An Example Document
  The first matching passage, which may start or stop mid-sentence...

  [2] Page 3 | An Example Document
  ...a further passage from the same page, continuing on from the first...

  [3] Page 11 | Another Document Entirely
  A passage from a different document.
  --- END LAYOUT ILLUSTRATION ---

Block numbering is not per search. It continues across every search you make, and across the whole conversation: a later search picks up where the last one stopped, and a block that comes back a second time keeps the number it already had. So a number identifies one passage for as long as the conversation lasts, and it is not reused for another — the reader is looking at the same numbered blocks you are.

You start every conversation holding no context whatsoever: until search_documents returns, you have seen no documents at all, and cannot say what they do or do not cover.

Documents are split into chunks before indexing, so a block may begin or end mid-sentence — this is expected. Multiple consecutive blocks from the same page often form one continuous passage; treat them together when both are returned. A block that appears incomplete does not mean the document lacks the information; a follow-up search with a refined query may surface adjacent chunks.

CITATION RULES
1. Always call search_documents before answering any factual question — including a question about what the documents are about, and including one you believe you already know the answer to.
2. Answer only from the returned context blocks. Never use knowledge from your training data.
3. Place the citation index in square brackets immediately after every factual claim — not at the end of the paragraph:
   Correct:   "The first measure rose [1] while the second fell [2]."
   Incorrect: "The first measure rose and the second fell. [1][2]"
4. If a claim draws from multiple blocks, list every relevant index: "Both measures moved together [1][3]."
5. You may quote directly when precision matters. When synthesising across blocks, paraphrase and cite all sources used.
6. Never fabricate page numbers, document titles, or passage content. Cite only blocks returned by a search you made while answering the current question: blocks from an earlier question are still visible above, but the reader is no longer shown them, so search again and cite the number that search returns.

WHEN CONTEXT IS INSUFFICIENT
7. If the returned blocks contain no relevant information, say exactly:
   "{REFUSAL_SENTENCE} [topic]."
   Do not speculate, infer, or substitute general knowledge. The absence of information in the chunks means you cannot answer that part of the question.
8. If the context is partially relevant — some aspects addressed, others not — answer the supported aspects with citations, then state explicitly:
   "{PARTIAL_REFUSAL_SENTENCE} [missing aspect]."
   Never silently skip the unanswered part.

FOLLOW-UP AND MULTI-PART QUESTIONS
9. For multi-part questions, call search_documents once per distinct information need when a single query would not surface all relevant chunks. Use a different, targeted query each time.
10. Do not describe your retrieval process or explain that you are calling a tool. Answer directly from the context as if it is naturally available to you."""

SEARCH_TOOL_DESCRIPTION = (
    "Search the loaded documents for information relevant to the query. "
    "Always call this before answering factual questions."
)

OCR_INSTRUCTION = (
    "Extract all text from this document page exactly as it appears. "
    "Return only the text, no commentary."
)

# Deliberately narrow: the judge is asked whether the answer *follows from* the
# passages, not whether it is true. A model asked for both falls back on what it
# already believes, and then it is scoring the corpus rather than the answer.
JUDGE_PROMPT = """You check whether an answer is supported by the source passages it cited.

You will be given a QUESTION, an ANSWER, and the numbered SOURCES the answer cited.

Judge only whether each claim in the ANSWER follows from the SOURCES. Do not judge
whether the claims are true in the wider world, and do not use anything you know
that is not in the SOURCES. An answer that is factually correct but not supported
by these passages is unsupported. An answer faithful to a passage that is itself
wrong is supported.

An answer that correctly declines to answer — saying the documents do not contain
the information — is fully supported. Score it 1.0.

Reply with JSON and nothing else:
{"faithfulness": <0.0 to 1.0>, "unsupported": ["<claim>", ...]}

faithfulness is the share of the answer's claims that the SOURCES support: 1.0 when
every claim is supported, 0.0 when none is. List each unsupported claim briefly in
"unsupported", using the answer's own words. Leave it empty when everything checks out."""

JUDGE_REQUEST = """QUESTION
{question}

ANSWER
{answer}

SOURCES
{sources}"""


# ── Confidence — `domain.confidence` ──────────────────────────────────────────

# The cosine band a retrieval reading is scaled across. Centred on the line
# `domain.statistics` already draws between a match worth having and one that is
# not, so a passage sitting exactly on it reads as 0.5 here and the two parts of
# the app cannot come to disagree about what "weak" means. The width is a
# calibration, not a measurement: below the floor is noise for every embedding
# model, above the ceiling is as good as retrieval gets on prose.
RETRIEVAL_FLOOR = WEAK_RETRIEVAL - 0.25
RETRIEVAL_CEILING = WEAK_RETRIEVAL + 0.25

# Retrieval and citations weigh equally: one says the corpus held an answer, the
# other says the reply is anchored to it, and an answer needs both. Completeness
# counts for half as much because it is the softest of the three — word overlap,
# not comprehension — and a confident number built mostly on it would be worth
# less than it looked.
RETRIEVAL_WEIGHT = 0.4
COVERAGE_WEIGHT = 0.4
COMPLETENESS_WEIGHT = 0.2

# Where a reader should stop trusting the number without opening the sources.
HIGH_CONFIDENCE = 0.75
MEDIUM_CONFIDENCE = 0.5

HIGH = "high"
MEDIUM = "medium"
LOW = "low"

# Under this many words a sentence is a heading, a lead-in, or a fragment — not a
# claim anyone would expect a citation after.
MIN_CLAIM_WORDS = 4
# How much of a question part's vocabulary has to turn up in the answer before it
# counts as addressed. Half, because an answer paraphrases: it restates the part
# it is answering in its own words, and demanding all of them would read every
# well-written answer as having ignored the question.
ADDRESSED_OVERLAP = 0.5

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
# Sentence ends, and line breaks with them: a bulleted list is a list of claims
# even when no line in it ends in a full stop.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
# Question parts. `?` ends one, `;` and a line break separate them.
_PART_SPLIT = re.compile(r"[?;\n]+")
# How a second question starts, in either of the two languages the folder is
# likely to hold. Only used behind a conjunction, below.
_CUE = (
    r"(?:what|which|who|whose|whom|how|why|when|where|do|does|did|is|are|was|"
    r"were|can|could|should|would|will|has|have|had|list|explain|describe|"
    r"summari[sz]e|compare|cosa|come|perch[eé]|quando|dove|chi|quale|quali|"
    r"quanto|quanti|elenca|spiega|descrivi|confronta)"
)
# The other half of a multi-part question, and the half most readers write:
# "What did revenue do and how did headcount change?" is one sentence with one
# question mark and two things being asked.
#
# Split on the conjunction only when what follows it opens a question. "Terms and
# conditions" keeps its shape; "and how did headcount change" does not. The
# alternative — splitting on every "and" — invents a part out of any coordinated
# noun and then reports the answer as having ignored it.
_SUBPART_SPLIT = re.compile(
    rf"(?:,\s*)?\b(?:and|or|e|oppure)\s+(?={_CUE}\b)|,\s+(?={_CUE}\b)",
    re.IGNORECASE,
)

# Function words carry no topic, so they say nothing about whether an answer went
# near a question part. English, plus the Italian equivalents: the corpus this
# runs over is whatever was dropped in the folder, and an unremoved function word
# inflates overlap rather than merely failing to help.
_STOPWORDS = frozenset(
    """
    a about after all also an and any are as at be because been but by can could
    did do does for from had has have how i if in into is it its just like may
    me might more most much must my no not of on one only or other our out over
    said same should since so some such than that the their them then there
    these they this those to too under until up use used using very was we were
    what when where which while who why will with within would you your
    a al alla alle agli ai anche che chi ci come con cosa cui da dal dalla dei
    del della delle dello di dove e ed gli ha hanno ho il in la le lo ma nel
    nella nelle non o per perche perché piu più qual quale quali quando quanto
    quello questa queste questi questo se sono sul sulla sulle su tra un una uno
    """.split()
)


# ── Text normalization — `domain.text_normalization` ──────────────────────────

# How far into a page furniture is looked for, counted in non-blank lines from
# either end. Two, because a head and a page number routinely share the band.
EDGE_LINES = 2

# A line has to head this share of pages to be treated as a running head. Well
# below half deliberately: journals and books alternate heads between verso and
# recto, so each side of the pair only ever reaches about 50% on its own, and a
# threshold at half catches one of the two at best.
RUNNING_HEAD_RATIO = 0.35
RUNNING_HEAD_MIN_PAGES = 3
RUNNING_HEAD_MAX_LEN = 90

# Detection needs a run of pages to have anything to detect against; below this
# a repeat is as likely to be coincidence as furniture.
MIN_PAGES_FOR_DETECTION = 4

# A line that was wrapped by the right margin is a full measure wide. Anything
# shorter ended for a reason of its own — a paragraph closing, a heading, a table
# cell — and keeps its break. The floor is what stops a table of short cells from
# being joined into prose, since there the whole page is short lines.
WRAP_WIDTH_RATIO = 0.75
WRAP_MIN_WIDTH = 45
WIDTH_PERCENTILE = 0.5

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}
_PUNCTUATION = {
    "−": "-",  # minus sign — how a typesetter sets a negative number
    "‐": "-", "‑": "-",  # hyphen, non-breaking hyphen
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
}
_REMOVED = (
    "­"  # soft hyphen — a break opportunity, never a character
    "​‌‍﻿"  # zero-width spaces and the byte-order mark
    "⃝⃞⃟⃠"  # combining enclosures, e.g. the ® of `R` + `⃝`
)
_CHARACTER_MAP: dict[int, str | None] = {
    **{ord(char): replacement for char, replacement in _LIGATURES.items()},
    **{ord(char): replacement for char, replacement in _PUNCTUATION.items()},
    **{ord(char): None for char in _REMOVED},
    # Control characters, which carry no text and break tokenisers that meet
    # them. Tab and newline are structure and stay.
    **{code: None for code in range(0x20) if code not in (0x09, 0x0A)},
    0x7F: None,
}

_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
}
_QUOTE_MAP = {ord(char): replacement for char, replacement in _QUOTES.items()}

# A word broken at the margin: lowercase, hyphen, line end, lowercase. The case
# test is what keeps `U.S.-\nbased` and a dash between clauses out of it.
_HYPHEN_BREAK_RE = re.compile(r"(?<=[a-zà-ÿ])-\n(?=[a-zà-ÿ])")

# A page number as its own line, with whatever the typesetter set around it.
_PAGE_NUMBER_RE = re.compile(r"^[\s\-–—\[\(]*(\d{1,4})[\s\-–—\]\)]*$")


# ── Titles — `domain.titles` ──────────────────────────────────────────────────

TITLE_MIN_LENGTH = 4
TITLE_MAX_LENGTH = 200

# Archive exports separate title from authors, journal, DOI and hash with a run
# of dashes: "Some Paper -- Author, A -- Journal, 2013 -- doi 10_1111 -- ...".
_SEPARATOR_RE = re.compile(r"\s+(?:-{2,}|—|–|\|)\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_LETTER_RE = re.compile(r"[^\W\d_]")

# Producers that stamp a placeholder into the PDF title field rather than
# leaving it empty.
_PLACEHOLDERS = ("untitled", "no title", "microsoft word", "unknown", "document1")
_DOCUMENT_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".rtf", ".tex", ".dvi", ".ps", ".indd", ".qxd",
)


# ── Index variants — `domain.variants` ────────────────────────────────────────

# Long enough to keep a model id readable, short enough that the qualified name
# stays inside the 255 characters Milvus allows a collection.
READABLE_MAX = 180
