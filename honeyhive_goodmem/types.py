"""Shared type definitions for the GoodMem + HoneyHive integration.

This module deliberately mirrors the option shapes used by the reference
GoodMem integration so behaviour is identical across frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Supported file extension -> MIME type. Mirrors the reference integration.
MIME_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "txt": "text/plain",
    "html": "text/html",
    "md": "text/markdown",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
    "doc": "application/msword",
    "docx": (
        "application/vnd.openxmlformats-officedocument." "wordprocessingml.document"
    ),
    "xls": "application/vnd.ms-excel",
    "xlsx": ("application/vnd.openxmlformats-officedocument." "spreadsheetml.sheet"),
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": (
        "application/vnd.openxmlformats-officedocument." "presentationml.presentation"
    ),
}


def get_mime_type(extension: str) -> Optional[str]:
    """Resolve a file extension to a MIME type.

    Args:
        extension: File extension (with or without leading dot, any case).

    Returns:
        The MIME type string, or ``None`` if unknown.
    """
    if not extension:
        return None
    return MIME_TYPES.get(extension.lower().lstrip("."))


@dataclass(frozen=True)
class GoodMemConfig:
    """Configuration for connecting to a GoodMem API instance.

    Attributes:
        base_url: Base URL of the GoodMem API server, for example
            ``https://api.goodmem.ai`` or ``https://localhost:8080``.
        api_key: API key sent as the ``X-API-Key`` header.
        verify_ssl: Whether to verify the server TLS certificate. Set to
            ``False`` for local self-signed deployments.
        timeout: HTTP request timeout in seconds.
    """

    base_url: str
    api_key: str
    verify_ssl: bool = True
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("GoodMem base_url is required")
        if not self.api_key:
            raise ValueError("GoodMem api_key is required")


class GoodMemError(Exception):
    """Raised when the GoodMem API returns a non-2xx response.

    Attributes:
        status: HTTP status code returned by the server.
        response_body: Parsed JSON body, or raw text when JSON parsing fails.
    """

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        response_body: Optional[object] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.response_body = response_body
