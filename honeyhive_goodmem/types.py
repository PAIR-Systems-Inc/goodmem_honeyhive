"""Shared type definitions for the GoodMem + HoneyHive integration.

This module deliberately mirrors the option shapes used by the reference
GoodMem integration so behaviour is identical across frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

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


class SecretStr:
    """A string that never renders its own value.

    ``repr()``, ``str()``, ``format()``, logging and ``json.dumps(...,
    default=str)`` all produce ``**********``; ``dataclasses.asdict()`` copies
    the wrapper, not the value. Only :meth:`get_secret_value` returns the
    string. It is not a ``str`` subclass, so handing the wrapper itself to an
    HTTP library raises instead of sending the mask as a credential.

    The interface matches ``pydantic.SecretStr``, and a ``pydantic.SecretStr``
    is accepted wherever this is.
    """

    __slots__ = ("_secret_value",)
    _secret_value: str
    _MASK = "**********"

    def __init__(self, value: str) -> None:
        if type(value) is not str:
            raise TypeError(
                f"SecretStr wraps a str, not {type(value).__name__}; the value "
                "is not shown"
            )
        object.__setattr__(self, "_secret_value", value)

    def get_secret_value(self) -> str:
        """Return the raw string. Call this only to hand it to a transport."""
        return self._secret_value

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SecretStr is immutable")

    def __repr__(self) -> str:
        return f"SecretStr('{self._MASK}')" if self._secret_value else "SecretStr('')"

    def __str__(self) -> str:
        return self._MASK if self._secret_value else ""

    def __format__(self, spec: str) -> str:
        return format(str(self), spec)

    def __len__(self) -> int:
        return len(self._secret_value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretStr):
            return self._secret_value == other._secret_value
        return NotImplemented

    def __hash__(self) -> int:
        return hash((SecretStr, self._secret_value))

    def __reduce__(self) -> tuple[type[SecretStr], tuple[str]]:
        return (SecretStr, (self._secret_value,))


def _as_secret(value: object) -> SecretStr:
    """Wrap an API key given as a str, this SecretStr or pydantic's."""
    if isinstance(value, SecretStr):
        return value
    reveal = getattr(value, "get_secret_value", None)
    raw = reveal() if callable(reveal) else value
    if not isinstance(raw, str):
        raise TypeError(
            f"GoodMem api_key must be a str or SecretStr, not {type(value).__name__}"
        )
    return SecretStr(str.__str__(raw))


@dataclass(frozen=True, init=False)
class GoodMemConfig:
    """Configuration for connecting to a GoodMem API instance.

    Attributes:
        base_url: Base URL of the GoodMem API server, for example
            ``https://api.goodmem.ai`` or ``https://localhost:8080``.
        api_key: API key sent as the ``X-API-Key`` header. Pass a ``str`` (or
            a :class:`SecretStr` / ``pydantic.SecretStr``); it is always
            stored as a :class:`SecretStr`, so ``repr()``, ``str()``,
            f-strings, logging, ``dataclasses.asdict()`` and HoneyHive's
            ``@trace`` input capture show ``**********``. Read the raw value
            with :meth:`get_api_key` or ``api_key.get_secret_value()``.
        verify_ssl: Whether to verify the server TLS certificate. Set to
            ``False`` for local self-signed deployments.
        timeout: HTTP request timeout in seconds.
    """

    base_url: str
    # Typed as what is stored, so ``config.api_key.get_secret_value()``
    # type-checks; the constructor below still accepts a plain ``str``.
    api_key: SecretStr
    verify_ssl: bool = True
    timeout: float = 30.0

    def __init__(
        self,
        base_url: str,
        api_key: Union[str, SecretStr],
        verify_ssl: bool = True,
        timeout: float = 30.0,
    ) -> None:
        if not base_url:
            raise ValueError("GoodMem base_url is required")
        secret = _as_secret(api_key)
        if not secret.get_secret_value():
            raise ValueError("GoodMem api_key is required")
        # Frozen dataclass: assign through object.__setattr__.
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "api_key", secret)
        object.__setattr__(self, "verify_ssl", verify_ssl)
        object.__setattr__(self, "timeout", timeout)

    def get_api_key(self) -> str:
        """Return the raw API key, for handing to an HTTP client."""
        return _as_secret(self.api_key).get_secret_value()


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
