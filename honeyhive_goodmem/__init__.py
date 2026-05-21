"""GoodMem integration for HoneyHive.

GoodMem is a memory layer for AI agents with support for semantic storage,
retrieval, and summarization. This module exposes GoodMem operations as
HoneyHive-traced Python methods that can be called directly or driven by
any agent framework that integrates with HoneyHive's tracer.

Public API:
    - :class:`GoodMemClient` — traced client wrapping the GoodMem REST API.
    - :class:`GoodMemConfig` — connection configuration dataclass.
    - :class:`GoodMemError` — raised by the transport layer on HTTP errors.
    - :func:`get_mime_type` — helper resolving file extensions to MIME types.
"""

from .client import GoodMemClient
from .types import GoodMemConfig, GoodMemError, MIME_TYPES, get_mime_type

__all__ = [
    "GoodMemClient",
    "GoodMemConfig",
    "GoodMemError",
    "MIME_TYPES",
    "get_mime_type",
]
