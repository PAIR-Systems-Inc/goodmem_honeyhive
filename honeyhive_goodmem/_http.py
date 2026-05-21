"""Minimal HTTP transport for the GoodMem REST API.

Uses :mod:`httpx` (already a HoneyHive dependency) so the integration
works out-of-the-box without any extra installs. Centralizes header
construction, base URL normalization, and error handling so the public
:class:`GoodMemClient` stays focused on operation semantics.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

import httpx

from .types import GoodMemConfig, GoodMemError


class _GoodMemTransport:
    """Lightweight wrapper around :mod:`httpx` for the GoodMem API."""

    def __init__(self, config: GoodMemConfig) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._api_key = config.api_key
        self._verify = config.verify_ssl
        self._timeout = config.timeout

    def _headers(self, extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
        headers = {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if extra:
            headers.update(extra)
        return headers

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        body: Any
        try:
            body = response.json()
        except ValueError:
            body = response.text or None
        if isinstance(body, dict):
            message = body.get("message") or body.get("error")
        else:
            message = None
        if not message:
            message = f"HTTP {response.status_code}: {response.reason_phrase}"
        raise GoodMemError(message, status=response.status_code, response_body=body)

    def request_json(
        self,
        method: str,
        path: str,
        body: Optional[object] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> Any:
        """Send a JSON request and return the parsed JSON response.

        Returns the response text when the server replies with a non-JSON
        content type (e.g. NDJSON streams).
        """
        url = f"{self._base_url}{path}"
        payload = None if body is None else json.dumps(body)
        with httpx.Client(verify=self._verify, timeout=self._timeout) as client:
            response = client.request(
                method,
                url,
                content=payload,
                headers=self._headers(extra_headers),
            )
        self._raise_for_status(response)
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        return response.text

    def request_text(
        self,
        method: str,
        path: str,
        body: Optional[object] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> str:
        """Send a request and return the raw text response.

        Used for NDJSON / SSE endpoints such as ``/v1/memories:retrieve``.
        """
        url = f"{self._base_url}{path}"
        payload = None if body is None else json.dumps(body)
        headers = {"Accept": "application/x-ndjson"}
        if extra_headers:
            headers.update(extra_headers)
        with httpx.Client(verify=self._verify, timeout=self._timeout) as client:
            response = client.request(
                method,
                url,
                content=payload,
                headers=self._headers(headers),
            )
        self._raise_for_status(response)
        return response.text
