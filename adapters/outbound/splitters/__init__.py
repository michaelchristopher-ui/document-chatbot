"""Implementations of `domain.chunking.SplitText` — recursive text splitting.

The one package here named for a domain callable rather than a port in
`ports.outbound`: `build_chunker` takes the split as a plain function, so there
is no protocol to point at — only a dependency on LangChain that has to live on
this side of the boundary.
"""
