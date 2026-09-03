from __future__ import annotations

from typing import Iterator

import fitz  # pymupdf

from adapters.outbound.document_parsers.constants import (
    COVER_ZONE,
    JPEG_QUALITY,
    RENDER_SCALE,
    SIZE_TOLERANCE,
)
from domain.models import ParsedPage
from domain.titles import is_usable, normalize


class PyMuPdfParser:
    """Implements `DocumentParser` for PDFs."""

    def __init__(self, render_scale: float = RENDER_SCALE, jpeg_quality: int = JPEG_QUALITY):
        self._render_scale = render_scale
        self._jpeg_quality = jpeg_quality

    def title(self, data: bytes) -> str:
        """The declared title, falling back to the cover line it is set in.

        Typesetters fill the metadata field far less reliably than they set a
        title in the largest type on page one, and producers that fill it with a
        placeholder are common enough that a title has to be read before it is
        believed.
        """
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            declared = normalize((doc.metadata or {}).get("title") or "")
            if is_usable(declared):
                return declared
            cover = self._cover_line(doc)
            return cover if is_usable(cover) else ""
        finally:
            doc.close()

    def _cover_line(self, doc: fitz.Document) -> str:
        """The largest-set text in the top half of page one, in reading order."""
        if doc.page_count == 0:
            return ""
        page = doc[0]
        limit = page.rect.y0 + page.rect.height * COVER_ZONE
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", ())
            for span in line.get("spans", ())
            if span["text"].strip() and span["bbox"][1] < limit
        ]
        if not spans:
            return ""

        # A title runs over several spans when it wraps or changes weight, and
        # rounding differences make those spans fractionally unequal in size.
        largest = max(span["size"] for span in spans)
        return normalize(
            " ".join(
                span["text"] for span in spans if span["size"] >= largest * SIZE_TOLERANCE
            )
        )

    def page_count(self, data: bytes) -> int:
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            return doc.page_count
        finally:
            doc.close()

    def pages(self, data: bytes) -> Iterator[ParsedPage]:
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            for number, page in enumerate(doc, start=1):
                yield ParsedPage(
                    number=number,
                    text=page.get_text(),
                    render_image=lambda number=number: self._render(data, number),
                )
        finally:
            doc.close()

    def _render(self, data: bytes, number: int) -> bytes:
        """Reopen rather than hold a page handle: a `fitz.Page` dies with its
        document, and a ParsedPage may be rendered after iteration has moved on.
        Only pages without a text layer pay this, and the OCR call dwarfs it.
        """
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            matrix = fitz.Matrix(self._render_scale, self._render_scale)
            pixmap = doc[number - 1].get_pixmap(matrix=matrix)
            return pixmap.tobytes("jpeg", jpg_quality=self._jpeg_quality)
        finally:
            doc.close()
