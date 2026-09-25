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

The value that is sent is never the caller's object. It is a new, exact
``str`` built from the data the check looked at, and it is checked again
before it is returned. The first version of this check read one thing and
returned another -- ``str(value)`` for a :class:`uuid.UUID` and
``value.lower()`` for a string, both of which a subclass can override -- so
a ``uuid.UUID`` subclass whose ``__str__`` returned ``"../spaces/<id>"``
still sent ``DELETE /v1/spaces/<id>`` from ``delete_memory``.
"""

from __future__ import annotations

import operator
import re
import uuid
from typing import Any, Optional, cast

from .types import GoodMemError

#: The canonical 8-4-4-4-12 form, either case. Matched with ``fullmatch``: a
#: ``$`` anchor would also accept a trailing newline.
_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

#: What :func:`require_uuid` may return: lower case only.
_CANONICAL = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

#: The slot :class:`uuid.UUID` stores its 128-bit value in. Reading it through
#: the descriptor ignores an ``int`` property a subclass defines over it.
_UUID_INT_SLOT = vars(uuid.UUID)["int"]


def _canonical(value: object) -> Optional[str]:
    """Build the lower-case UUID text for ``value``, or ``None``.

    Only ``uuid.UUID``'s own slot and ``str``'s own methods are used, so
    nothing a subclass overrides (``__str__``, ``lower``, ``int``, ``hex``,
    ``__format__``) can change the result, and the type is taken from
    ``type(value)``, which a ``__class__`` property cannot fake.
    """
    kind = type(value)
    try:
        if issubclass(kind, uuid.UUID):
            number = operator.index(_UUID_INT_SLOT.__get__(value, uuid.UUID))
            return str(uuid.UUID(int=number))
        if issubclass(kind, str):
            text = cast(str, value)
            return str.lower(text) if _UUID.fullmatch(text) else None
    except Exception:
        # An uninitialised UUID, a stored value that is not a 128-bit int, or
        # one whose __index__ raises: whatever it is, it is not an id we can
        # vouch for, and the caller gets GoodMemError rather than the value's
        # own exception.
        return None
    return None


def _shown(value: object) -> str:
    """The caller's value for an error message, without trusting its repr."""
    try:
        if issubclass(type(value), str):
            shown = str.__repr__(cast(str, value))
        else:
            shown = repr(value)
    except Exception:
        shown = f"<{type(value).__name__} object>"
    return shown if len(shown) <= 80 else shown[:77] + "..."


def require_uuid(value: object, field: str) -> str:
    """Return ``value`` as a lower-case UUID string, or refuse it.

    Args:
        value: The id the caller supplied. A :class:`uuid.UUID` is accepted,
            and so is a subclass of ``str`` or ``uuid.UUID`` -- but only its
            underlying data is used, never a method it overrides.
        field: The argument name, used in the error message.

    Returns:
        A new, exact ``str`` holding the canonical, lower-case form of the id.

    Raises:
        GoodMemError: If ``value`` is not a canonical UUID. No request is
            made.
    """
    text = _canonical(value)
    # The invariant every caller relies on, checked on what is returned
    # rather than on what came in.
    if type(text) is str and _CANONICAL.fullmatch(text):
        return text
    raise GoodMemError(
        f"{field} must be a UUID (GoodMem ids are UUIDs, e.g. "
        f"01a0d44b-748d-72eb-b54e-c3ea2d956927); got {_shown(value)}. "
        "Nothing was sent."
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
    if values is None or issubclass(type(values), (str, bytes, uuid.UUID)):
        return [require_uuid(values, field)]
    try:
        items = list(values)
    except TypeError:
        return [require_uuid(values, field)]
    except Exception as error:
        # An iterable whose __iter__/__next__ raises something else: refuse it
        # through the normal error type, before any request is made.
        raise GoodMemError(
            f"{field} could not be read as a list of ids ({type(error).__name__})."
        ) from error
    return [require_uuid(value, f"{field}[{i}]") for i, value in enumerate(items)]
