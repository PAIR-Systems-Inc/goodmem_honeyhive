"""The one place a GoodMem id is checked before it is sent.

The GoodMem SDK builds URL paths as f-strings -- ``f"/v1/memories/{id}"`` --
with the id interpolated raw, and httpx resolves dot segments before sending.
On 0.2.0, ``delete_memory("../spaces/<id>")`` therefore sent
``DELETE /v1/spaces/<id>``, deleted a whole space and returned
``success: True``. The server also normalises ``%2e%2e`` into a traversal, so
neither the client's encoding nor the server can be relied on.

Every GoodMem id is a UUID, so every id a caller hands to
:class:`~honeyhive_goodmem.GoodMemClient` goes through :func:`require_uuid`
before any request is made, and anything else is refused.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from .types import GoodMemError

#: The canonical 8-4-4-4-12 form. Matched with ``fullmatch``: a ``$`` anchor
#: would also accept a trailing newline.
_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def require_uuid(value: object, field: str) -> str:
    """Return ``value`` as a lower-case UUID string, or refuse it.

    Args:
        value: The id the caller supplied. A :class:`uuid.UUID` is accepted.
        field: The argument name, used in the error message.

    Returns:
        The canonical, lower-case form of the id.

    Raises:
        GoodMemError: If ``value`` is not a canonical UUID. No request is
            made.
    """
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, str) and _UUID.fullmatch(value):
        return value.lower()
    shown = repr(value)
    if len(shown) > 80:
        shown = shown[:77] + "..."
    raise GoodMemError(
        f"{field} must be a UUID (GoodMem ids are UUIDs, e.g. "
        f"01a0d44b-748d-72eb-b54e-c3ea2d956927); got {shown}. Nothing was sent."
    )


def require_uuids(values: Any, field: str) -> list[str]:
    """Check one id or several with :func:`require_uuid`.

    Args:
        values: A single id, or an iterable of ids.
        field: The argument name, used in the error message; an id from an
            iterable is named ``field[index]``.

    Returns:
        The canonical, lower-case ids, in order.

    Raises:
        GoodMemError: If any of them is not a canonical UUID.
    """
    if values is None or isinstance(values, (str, bytes, uuid.UUID)):
        return [require_uuid(values, field)]
    try:
        items = list(values)
    except TypeError:
        return [require_uuid(values, field)]
    return [require_uuid(value, f"{field}[{i}]") for i, value in enumerate(items)]
