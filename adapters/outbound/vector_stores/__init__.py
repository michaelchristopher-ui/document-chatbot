"""Implementations of `ports.outbound.VectorStore`, and the choice between them.

The port's docstring carries the contract every module here has to honour:
cosine similarity where larger is closer, `Passage.score` always set, and
`where` scoping that agrees with `MetadataFilter.matches`. `registry` is the one
place a backend is named.
"""
