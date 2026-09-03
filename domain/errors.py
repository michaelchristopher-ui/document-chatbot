from __future__ import annotations


class ChatbotError(Exception):
    """Base class for errors the application layer raises deliberately."""


class BackendUnavailable(ChatbotError):
    """An outbound adapter could not reach the service behind it."""

    def __init__(self, backend: str, endpoint: str, detail: str):
        self.backend = backend
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"{backend} unavailable at {endpoint}: {detail}")


class NoDocumentsFound(ChatbotError):
    """The document repository is empty, so there is nothing to ingest."""

    def __init__(self, location: str):
        self.location = location
        super().__init__(f"No documents found in {location}")
