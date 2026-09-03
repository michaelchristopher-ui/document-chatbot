from __future__ import annotations

import glob
import os

from domain.models import DocumentRef


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


class FilesystemDocuments:
    """Implements `DocumentRepository` over a local directory."""

    def __init__(self, directory: str, pattern: str = "*.pdf"):
        self._directory = directory
        self._pattern = pattern

    @property
    def location(self) -> str:
        return self._directory

    def list_documents(self) -> list[DocumentRef]:
        paths = sorted(glob.glob(os.path.join(self._directory, self._pattern)))
        return [
            DocumentRef(name=os.path.basename(path), read=lambda path=path: _read(path))
            for path in paths
        ]
