"""Timings and wiring for the registry-backed prompt library."""

from __future__ import annotations

# How long a resolved body is reused before the registry is asked again. This is
# the whole latency story: `text` is called on the answer path, several times per
# turn, and without a cache each of those would be an HTTP round trip before the
# model was even asked.
#
# It is also the lag. A prompt published in the console reaches a running app
# somewhere inside this window, so it is a trade between how fast an edit takes
# effect and how often a turn waits on the registry. Five minutes is short enough
# to iterate on a prompt without restarting anything, and long enough that a
# conversation costs at most a couple of lookups per key.
DEFAULT_TTL_SECONDS = 300.0

# A failed lookup is remembered for a shorter time than a successful one, and for
# a plain reason: a miss costs the app nothing — it answers on the built-in
# prompt — while a registry that has just come back up, or a key that has just
# been published for the first time, should be noticed sooner than five minutes
# later. Short enough to recover promptly, long enough not to retry a dead
# server on every turn.
FAILURE_TTL_SECONDS = 30.0

# Longer than the registry client's own 5s default would be a turn visibly
# waiting on a prompt store. Two seconds is enough for a service on this network
# and short enough that an unreachable one is a hiccup rather than a hang.
REQUEST_TIMEOUT_SECONDS = 2.0
