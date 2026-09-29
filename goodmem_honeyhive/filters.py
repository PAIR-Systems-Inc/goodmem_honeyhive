"""Public helpers for building GoodMem metadata filter expressions.

Example::

    from goodmem_honeyhive import filters

    expression = filters.all_of(
        filters.equals("tenant", "acme"),
        filters.compare("year", ">=", 2026),
    )
"""

from goodmem_honeyhive._filters import (
    GoodMemFilterError,
    all_of,
    any_of,
    compare,
    equals,
    escape_literal,
    from_mapping,
    not_equals,
    one_of,
)

__all__ = [
    "GoodMemFilterError",
    "all_of",
    "any_of",
    "compare",
    "equals",
    "escape_literal",
    "from_mapping",
    "not_equals",
    "one_of",
]
