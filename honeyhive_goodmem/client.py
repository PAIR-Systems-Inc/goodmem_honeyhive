"""GoodMem client exposed as HoneyHive-traced operations.

Every operation is decorated with HoneyHive's ``@trace`` so calls show up as
spans in the active session. That is the point of this package, and it is why
0.1.0's dropped retrieval statuses mattered more here than anywhere else: a
degraded retrieval was recorded as a *successful* span, so the observability
tool reported success for a search that had failed.

Retrieval now carries ``partial`` and ``statuses`` in the traced payload, so
a span shows what the server actually reported.

Every id argument must be a UUID and is checked before any request is made:
the SDK puts ids into URL paths raw, so ``delete_memory("../spaces/<id>")``
used to delete a whole space. A refused id raises :class:`GoodMemError`.

Example::

    from honeyhive import HoneyHiveTracer
    from honeyhive_goodmem import GoodMemClient, GoodMemConfig

    HoneyHiveTracer.init(api_key="...", project="...")
    client = GoodMemClient(GoodMemConfig(base_url="...", api_key="..."))
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Optional, Union

from honeyhive import trace

from ._filters import from_mapping
from ._ids import require_uuid, require_uuids
from ._results import (
    RetrievalOutcome,
    log_if_degraded,
    outcome_from_events,
)
from .types import MIME_TYPES, GoodMemConfig, GoodMemError

logger = logging.getLogger(__name__)

DEFAULT_MAX_LIST_ITEMS = 200


def _wrap(exc: Exception, what: str) -> GoodMemError:
    """Convert an SDK error into a GoodMemError, keeping the server's body."""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    detail = str(exc)
    if body and body not in detail:
        detail = f"{detail} -- {body}"
    error = GoodMemError(f"{what} failed: {detail}")
    error.status_code = status  # type: ignore[attr-defined]
    error.body = body  # type: ignore[attr-defined]
    return error


def _space_embedder_ids(space: Any) -> list[str]:
    """Return the embedder ids a space is actually indexed by."""
    out: list[str] = []
    for config in getattr(space, "space_embedders", None) or []:
        value = getattr(config, "embedder_id", None)
        if value is None and isinstance(config, dict):
            value = config.get("embedderId") or config.get("embedder_id")
        if value:
            out.append(str(value))
    return out


def _decode_content(raw: bytes, content_type: str) -> tuple[Any, str]:
    """Decode memory content by its content type: text as text, else base64."""
    primary = (content_type or "").split(";")[0].strip().lower()
    charset = "utf-8"
    for part in (content_type or "").split(";")[1:]:
        if "charset=" in part:
            charset = part.split("charset=", 1)[1].strip() or "utf-8"
    textual = primary.startswith("text/") or primary in {
        "application/json",
        "application/xml",
        "application/javascript",
    }
    if textual:
        try:
            return raw.decode(charset), "text"
        except (UnicodeDecodeError, LookupError):
            return base64.b64encode(raw).decode("ascii"), "base64"
    return base64.b64encode(raw).decode("ascii"), "base64"


class GoodMemClient:
    """A HoneyHive-traced client for GoodMem.

    Args:
        config: Connection settings. ``base_url`` and ``api_key`` fall back
            to ``GOODMEM_BASE_URL`` and ``GOODMEM_API_KEY``.
        client: An already-configured ``goodmem.Goodmem`` client. When given,
            its server, credentials and TLS settings are used as-is and it is
            never closed here.
        max_list_items: Upper bound on items returned by a listing.
    """

    def __init__(
        self,
        config: Optional[GoodMemConfig] = None,
        *,
        client: Any = None,
        max_list_items: int = DEFAULT_MAX_LIST_ITEMS,
    ) -> None:
        from goodmem import Goodmem

        config = config or GoodMemConfig(
            base_url=os.environ.get("GOODMEM_BASE_URL", ""),
            api_key=os.environ.get("GOODMEM_API_KEY", ""),
        )
        self.base_url = (config.base_url or "").rstrip("/")
        # The key is not kept here at all: it goes straight to the SDK client.
        # GoodMemConfig holds it as a SecretStr, so a config that reaches a
        # repr, a log line or HoneyHive's @trace input capture shows a mask.
        api_key = config.get_api_key()
        self.verify_ssl = config.verify_ssl
        self.max_list_items = max_list_items

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            missing = [
                name
                for name, value in (
                    ("GOODMEM_API_KEY", api_key),
                    ("GOODMEM_BASE_URL", self.base_url),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    f"Missing GoodMem credentials: {', '.join(missing)}. Set "
                    "them in the environment or pass a GoodMemConfig, or pass "
                    "an already-configured client=Goodmem(...)."
                )
            self._client = Goodmem(
                base_url=self.base_url,
                api_key=api_key,
                timeout=config.timeout,
                verify=config.verify_ssl,
            )
            self._owns_client = True

    def __repr__(self) -> str:
        """A representation that never carries the API key."""
        return f"{type(self).__name__}(base_url={self.base_url!r})"

    def close(self) -> None:
        """Close the HTTP client, if this object created it."""
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> GoodMemClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # spaces
    # ------------------------------------------------------------------

    @trace(event_type="tool", event_name="goodmem.create_space")
    def create_space(self, name: str, embedder_id: str) -> dict[str, Any]:
        """Create a space, or reuse one whose embedder already matches.

        A space cannot change embedder after creation, so reusing by name
        alone silently writes vectors from a different model than the caller
        asked for -- and 0.1.0 reported the embedder you *asked* for while
        the space ran another.

        Args:
            name: The space name.
            embedder_id: The embedder the space must use.

        Returns:
            ``success``, ``space_id``, ``name``, ``embedder_id``, ``reused``.

        Raises:
            GoodMemError: If ``embedder_id`` is not a UUID, if a space of
                that name exists with a different embedder, or if several
                spaces share the name.
        """
        embedder_id = require_uuid(embedder_id, "embedder_id")
        existing = [
            s for s in self._list_spaces_raw() if str(getattr(s, "name", "")) == name
        ]
        if len(existing) > 1:
            raise GoodMemError(
                f"{len(existing)} spaces are named {name!r}; refusing to guess "
                "which one was meant. Pass a space id instead."
            )
        if existing:
            space = existing[0]
            actual = _space_embedder_ids(space)
            if embedder_id not in actual:
                raise GoodMemError(
                    f"Space {name!r} already exists and is indexed by embedder(s) "
                    f"{actual}, not {embedder_id!r}. An embedder cannot be changed "
                    "after creation."
                )
            return {
                "success": True,
                "space_id": str(getattr(space, "space_id", "")),
                "name": name,
                "embedder_id": embedder_id,
                "reused": True,
            }
        try:
            space = self._client.spaces.create(
                name=name,
                space_embedders=[
                    {"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}
                ],
            )
        except Exception as exc:
            raise _wrap(exc, f"Creating space {name!r}") from exc
        return {
            "success": True,
            "space_id": str(getattr(space, "space_id", "")),
            "name": str(getattr(space, "name", "") or name),
            "embedder_id": embedder_id,
            "reused": False,
        }

    def _list_spaces_raw(self) -> list[Any]:
        try:
            return list(self._client.spaces.list(max_items=self.max_list_items))
        except Exception as exc:
            raise _wrap(exc, "Listing spaces") from exc

    @trace(event_type="tool", event_name="goodmem.list_spaces")
    def list_spaces(self) -> dict[str, Any]:
        """List spaces, following pagination up to ``max_list_items``."""
        spaces = self._list_spaces_raw()
        return {
            "success": True,
            "spaces": [
                {
                    "space_id": str(getattr(s, "space_id", "")),
                    "name": str(getattr(s, "name", "")),
                    "embedder_ids": _space_embedder_ids(s),
                }
                for s in spaces
            ],
            "total_results": len(spaces),
        }

    @trace(event_type="tool", event_name="goodmem.get_space")
    def get_space(self, space_id: str) -> dict[str, Any]:
        """Fetch one space by id."""
        space_id = require_uuid(space_id, "space_id")
        try:
            space = self._client.spaces.get(id=space_id)
        except Exception as exc:
            raise _wrap(exc, f"Fetching space {space_id}") from exc
        return {
            "success": True,
            "space_id": str(getattr(space, "space_id", "")),
            "name": str(getattr(space, "name", "")),
            "embedder_ids": _space_embedder_ids(space),
            "labels": dict(getattr(space, "labels", None) or {}),
        }

    @trace(event_type="tool", event_name="goodmem.update_space")
    def update_space(
        self,
        space_id: str,
        name: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        replace_labels: bool = False,
    ) -> dict[str, Any]:
        """Rename a space or edit its labels.

        ``public_read`` is deliberately gone: the server removed the field
        and answers ``400 Unrecognized field "publicRead"``.
        """
        space_id = require_uuid(space_id, "space_id")
        request: dict[str, Any] = {}
        if name is not None:
            request["name"] = name
        if labels is not None:
            request["replaceLabels" if replace_labels else "mergeLabels"] = dict(labels)
        if not request:
            raise GoodMemError("update_space() needs a name or labels to change.")
        try:
            space = self._client.spaces.update(id=space_id, request=request)
        except Exception as exc:
            raise _wrap(exc, f"Updating space {space_id}") from exc
        return {
            "success": True,
            "space_id": str(getattr(space, "space_id", "") or space_id),
            "name": str(getattr(space, "name", "")),
        }

    @trace(event_type="tool", event_name="goodmem.delete_space")
    def delete_space(self, space_id: str) -> dict[str, Any]:
        """Permanently delete a space and every memory in it."""
        space_id = require_uuid(space_id, "space_id")
        try:
            self._client.spaces.delete(id=space_id)
        except Exception as exc:
            raise _wrap(exc, f"Deleting space {space_id}") from exc
        return {"success": True, "space_id": space_id}

    @trace(event_type="tool", event_name="goodmem.list_embedders")
    def list_embedders(self) -> dict[str, Any]:
        """List the embedder models available on the server."""
        try:
            embedders = list(self._client.embedders.list())
        except Exception as exc:
            raise _wrap(exc, "Listing embedders") from exc
        return {
            "success": True,
            "embedders": [
                {
                    "embedder_id": str(getattr(e, "embedder_id", "")),
                    "display_name": str(getattr(e, "display_name", "")),
                    "model_identifier": str(getattr(e, "model_identifier", "")),
                }
                for e in embedders
            ],
            "total_results": len(embedders),
        }

    # ------------------------------------------------------------------
    # memories
    # ------------------------------------------------------------------

    @trace(event_type="tool", event_name="goodmem.create_memory")
    def create_memory(
        self,
        space_id: str,
        file_base64: Optional[str] = None,
        file_extension: Optional[str] = None,
        text_content: Optional[str] = None,
        source: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[Union[str, list[str]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Store a document or text snippet as a memory.

        Args:
            space_id: Target space.
            file_base64: Base64-encoded file content. The MIME type comes
                from ``file_extension``.
            file_extension: Extension used to pick the MIME type.
            text_content: Plain text. Ignored when ``file_base64`` is given.
            source: Where the memory came from.
            author: Who wrote it.
            tags: One tag or several.
            metadata: Extra key-value labels.

        Returns:
            ``success``, ``memory_id``, ``space_id``, ``status``,
            ``content_type``.
        """
        space_id = require_uuid(space_id, "space_id")
        meta: dict[str, Any] = dict(metadata or {})
        if source:
            meta["source"] = source
        if author:
            meta["author"] = author
        if tags:
            meta["tags"] = ",".join(tags) if isinstance(tags, list) else tags

        kwargs: dict[str, Any] = {"space_id": space_id, "metadata": meta or None}
        if file_base64:
            ext = (file_extension or "").lower().lstrip(".")
            content_type = MIME_TYPES.get(ext, "application/octet-stream")
            kwargs["original_content_b64"] = file_base64
            kwargs["content_type"] = content_type
        elif text_content:
            content_type = "text/plain"
            kwargs["original_content"] = text_content
            kwargs["content_type"] = content_type
        else:
            raise GoodMemError("Provide file_base64 or text_content.")

        try:
            memory = self._client.memories.create(**kwargs)
        except Exception as exc:
            raise _wrap(exc, "Creating a memory") from exc
        return {
            "success": True,
            "memory_id": str(getattr(memory, "memory_id", "")),
            "space_id": str(getattr(memory, "space_id", "") or space_id),
            "status": str(getattr(memory, "processing_status", "")),
            "content_type": content_type,
        }

    @trace(event_type="tool", event_name="goodmem.get_memory")
    def get_memory(
        self, memory_id: str, include_content: bool = False
    ) -> dict[str, Any]:
        """Fetch one memory, optionally with its original content.

        Content is decoded by the memory's own content type: text as text,
        anything else as base64, so the traced payload is always
        JSON-serialisable. A content fetch that fails is an error, not a
        successful result with a note in it.
        """
        memory_id = require_uuid(memory_id, "memory_id")
        try:
            memory = self._client.memories.get(id=memory_id)
        except Exception as exc:
            raise _wrap(exc, f"Fetching memory {memory_id}") from exc
        dump = getattr(memory, "model_dump", None)
        payload = dump(by_alias=True, exclude_none=True) if dump else {}
        result: dict[str, Any] = {"success": True, "memory": payload}
        if include_content:
            try:
                raw = self._client.memories.content(id=memory_id)
            except Exception as exc:
                raise _wrap(exc, f"Fetching content of memory {memory_id}") from exc
            content_type = str(
                payload.get("contentType") or payload.get("content_type") or ""
            )
            result["content"], result["content_encoding"] = _decode_content(
                raw, content_type
            )
        return result

    @trace(event_type="tool", event_name="goodmem.list_memories")
    def list_memories(self, space_id: str) -> dict[str, Any]:
        """List memories in a space, following pagination."""
        space_id = require_uuid(space_id, "space_id")
        try:
            memories = list(
                self._client.memories.list(
                    space_id=space_id, max_items=self.max_list_items
                )
            )
        except Exception as exc:
            raise _wrap(exc, "Listing memories") from exc
        return {
            "success": True,
            "memories": [
                {
                    "memory_id": str(getattr(m, "memory_id", "")),
                    "space_id": str(getattr(m, "space_id", "")),
                    "content_type": str(getattr(m, "content_type", "")),
                    "processing_status": str(getattr(m, "processing_status", "")),
                    "metadata": dict(getattr(m, "metadata", None) or {}),
                }
                for m in memories
            ],
            "total_results": len(memories),
        }

    @trace(event_type="tool", event_name="goodmem.delete_memory")
    def delete_memory(self, memory_id: str) -> dict[str, Any]:
        """Permanently delete a memory and everything derived from it."""
        memory_id = require_uuid(memory_id, "memory_id")
        try:
            self._client.memories.delete(id=memory_id)
        except Exception as exc:
            raise _wrap(exc, f"Deleting memory {memory_id}") from exc
        return {"success": True, "memory_id": memory_id}

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        space_ids: Union[str, list[str]],
        *,
        max_results: int = 5,
        reranker_id: Optional[str] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> RetrievalOutcome:
        """Retrieve chunks relevant to a query, as a structured outcome."""
        ids = require_uuids(space_ids, "space_ids")
        if not ids:
            raise GoodMemError("At least one space id is required.")
        if reranker_id is not None:
            reranker_id = require_uuid(reranker_id, "reranker_id")
        expression = from_mapping(metadata_filter or {})
        keys: list[dict[str, Any]] = []
        for space_id in ids:
            key: dict[str, Any] = {"spaceId": space_id}
            if expression:
                key["filter"] = expression
            keys.append(key)
        kwargs: dict[str, Any] = {
            "message": query,
            "space_keys": keys,
            "requested_size": max_results,
            "fetch_memory": True,
        }
        if reranker_id:
            kwargs["reranker_id"] = reranker_id
        try:
            stream = self._client.memories.retrieve(**kwargs)
            with stream as events:
                outcome = outcome_from_events(events, reranked=bool(reranker_id))
        except GoodMemError:
            raise
        except Exception as exc:
            raise _wrap(exc, "Retrieval") from exc
        log_if_degraded(outcome, "goodmem.retrieve_memories")
        return outcome

    @trace(event_type="retrieval", event_name="goodmem.retrieve_memories")
    def retrieve_memories(
        self,
        query: str,
        space_ids: Union[str, list[str]],
        max_results: int = 5,
        reranker_id: Optional[str] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Retrieve memories, as a traced payload.

        The returned dictionary is what a HoneyHive span records. It carries
        ``partial`` and ``statuses``, so a degraded retrieval is visible in
        the trace instead of being recorded as a clean success.

        Args:
            query: The natural-language query.
            space_ids: One space id or several. Each must be a UUID.
            max_results: How many chunks to ask the server for.
            reranker_id: A reranker to apply, or ``None`` for none. Must be a
                UUID; an empty string is refused, not read as ``None``.
            metadata_filter: Metadata every memory must match, applied
                server-side and escaped by :mod:`honeyhive_goodmem.filters`.

        Returns:
            ``success``, ``query``, ``results``, ``total_results``,
            ``partial``, ``statuses``, ``result_set_id`` and, when degraded,
            ``warning``.
        """
        outcome = self.retrieve(
            query,
            space_ids,
            max_results=max_results,
            reranker_id=reranker_id,
            metadata_filter=metadata_filter,
        )
        payload: dict[str, Any] = {
            "success": True,
            "query": query,
            "results": [
                {
                    "chunk_id": h.chunk_id,
                    "chunk_text": h.text,
                    "memory_id": h.memory_id,
                    "space_id": h.space_id,
                    "score": h.score,
                    "raw_score": h.raw_score,
                    "score_kind": h.score_kind,
                    "content_type": h.content_type,
                    "metadata": h.metadata,
                }
                for h in outcome.hits
            ],
            "total_results": len(outcome.hits),
            "partial": outcome.partial,
            "statuses": outcome.status_dicts,
            "result_set_id": outcome.result_set_id,
        }
        if outcome.partial:
            payload["warning"] = outcome.warning_text()
        if outcome.abstract_reply:
            payload["abstract_reply"] = outcome.abstract_reply
        return payload
