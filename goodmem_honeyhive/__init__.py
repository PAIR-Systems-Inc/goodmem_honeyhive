"""GoodMem integration for HoneyHive.

Exposes GoodMem operations as HoneyHive-traced methods, so memory reads and
writes appear as spans alongside the rest of an agent's work -- and a
degraded retrieval is visible in the trace rather than recorded as a clean
success.
"""

from goodmem_honeyhive import filters
from goodmem_honeyhive._filters import GoodMemFilterError
from goodmem_honeyhive._results import (
    INFORMATIONAL_CODES,
    MALFORMED_STREAM_CODE,
    UNKNOWN_CODE,
    RetrievalHit,
    RetrievalOutcome,
    RetrievalStatus,
)
from goodmem_honeyhive.client import GoodMemClient
from goodmem_honeyhive.types import GoodMemConfig, GoodMemError, SecretStr

__version__ = "0.4.0"

__all__ = [
    "GoodMemClient",
    "GoodMemConfig",
    "GoodMemError",
    "GoodMemFilterError",
    "SecretStr",
    "RetrievalHit",
    "RetrievalOutcome",
    "RetrievalStatus",
    "INFORMATIONAL_CODES",
    "MALFORMED_STREAM_CODE",
    "UNKNOWN_CODE",
    "filters",
    "__version__",
]
