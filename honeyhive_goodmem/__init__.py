"""GoodMem integration for HoneyHive.

Exposes GoodMem operations as HoneyHive-traced methods, so memory reads and
writes appear as spans alongside the rest of an agent's work -- and a
degraded retrieval is visible in the trace rather than recorded as a clean
success.
"""

from honeyhive_goodmem import filters
from honeyhive_goodmem._filters import GoodMemFilterError
from honeyhive_goodmem._results import (
    INFORMATIONAL_CODES,
    MALFORMED_STREAM_CODE,
    UNKNOWN_CODE,
    RetrievalHit,
    RetrievalOutcome,
    RetrievalStatus,
)
from honeyhive_goodmem.client import GoodMemClient
from honeyhive_goodmem.types import GoodMemConfig, GoodMemError

__version__ = "0.2.0"

__all__ = [
    "GoodMemClient",
    "GoodMemConfig",
    "GoodMemError",
    "GoodMemFilterError",
    "RetrievalHit",
    "RetrievalOutcome",
    "RetrievalStatus",
    "INFORMATIONAL_CODES",
    "MALFORMED_STREAM_CODE",
    "UNKNOWN_CODE",
    "filters",
    "__version__",
]
