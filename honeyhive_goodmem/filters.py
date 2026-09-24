"""Public helpers for building GoodMem metadata filter expressions.

Example::

    from honeyhive_goodmem import filters

    expression = filters.all_of(
        filters.equals("tenant", "acme"),
        filters.compare("year", ">=", 2026),
    )
"""

from honeyhive_goodmem._filters import (
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
