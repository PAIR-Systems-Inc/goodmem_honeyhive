"""GoodMem client exposed as HoneyHive-traced operations.

Each public method on :class:`GoodMemClient` is decorated with HoneyHive's
``@trace`` so calls show up as spans in the active HoneyHive session. The
operations and their option shapes mirror the reference GoodMem integration
verbatim — only the surface (Python methods returning dicts vs. agent tools
returning JSON) is adapted to match honeyhive conventions.

Usage::

    from honeyhive import HoneyHiveTracer

    from goodmem.client import GoodMemClient
    from goodmem.types import GoodMemConfig

    HoneyHiveTracer.init(api_key="...", project="...")

    client = GoodMemClient(GoodMemConfig(base_url="...", api_key="..."))
    spaces = client.list_spaces()
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Optional, Union

from honeyhive import trace

from ._http import _GoodMemTransport
from .types import GoodMemConfig, GoodMemError, get_mime_type

# Defaults match the reference integration so behaviour is identical.
_DEFAULT_CHUNK_SIZE = 256
_DEFAULT_CHUNK_OVERLAP = 25
_DEFAULT_KEEP_STRATEGY = "KEEP_END"
_DEFAULT_LENGTH_MEASUREMENT = "CHARACTER_COUNT"
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_DEFAULT_MAX_RESULTS = 5
_RETRIEVE_MAX_WAIT_SECONDS = 10.0
_RETRIEVE_POLL_INTERVAL_SECONDS = 2.0


def _error_result(operation: str, error: BaseException) -> dict[str, Any]:
    """Build a uniform failure result for any GoodMem operation."""
    if isinstance(error, GoodMemError):
        message = str(error) or f"Failed to {operation}"
        return {
            "success": False,
            "error": message,
            "details": error.response_body,
        }
    return {
        "success": False,
        "error": str(error) or f"Failed to {operation}",
        "details": None,
    }


class GoodMemClient:
    """Public client for the GoodMem memory-layer API.

    Every method is traced through HoneyHive when a tracer has been
    initialised via :func:`HoneyHiveTracer.init`. When no tracer is active
    the methods still execute normally — the ``@trace`` decorator degrades
    gracefully.

    Args:
        config: GoodMem connection configuration.
    """

    def __init__(self, config: GoodMemConfig) -> None:
        self._config = config
        self._transport = _GoodMemTransport(config)

    # ------------------------------------------------------------------ #
    # Create Space
    # ------------------------------------------------------------------ #
    @trace(event_type="tool", event_name="goodmem.create_space")
    def create_space(
        self,
        name: str,
        embedder_id: str,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        keep_strategy: str = _DEFAULT_KEEP_STRATEGY,
        length_measurement: str = _DEFAULT_LENGTH_MEASUREMENT,
    ) -> dict[str, Any]:
        """Create a new space, or reuse one with the same name.

        Args:
            name: Unique name for the space. If a space with this name
                already exists, its ID is returned instead of creating a
                duplicate.
            embedder_id: Embedder model ID. Use :meth:`list_embedders` to
                discover available embedders.
            chunk_size: Characters per chunk when splitting documents.
            chunk_overlap: Overlapping characters between consecutive chunks.
            keep_strategy: Where to attach the separator when splitting.
                One of ``"KEEP_END"``, ``"KEEP_START"``, ``"DISCARD"``.
            length_measurement: How chunk size is measured. One of
                ``"CHARACTER_COUNT"`` or ``"TOKEN_COUNT"``.

        Returns:
            ``{success, space_id, name, embedder_id, message, reused, ...}``.
            On failure ``{success: False, error, details}``.
        """
        # Reuse the existing space when the name already exists.
        try:
            existing_body = self._transport.request_json("GET", "/v1/spaces")
            spaces = (
                existing_body
                if isinstance(existing_body, list)
                else (existing_body or {}).get("spaces") or []
            )
            for space in spaces:
                if space.get("name") == name:
                    return {
                        "success": True,
                        "space_id": space.get("spaceId") or space.get("id"),
                        "name": space.get("name"),
                        "embedder_id": embedder_id,
                        "message": "Space already exists, reusing existing space",
                        "reused": True,
                    }
        except GoodMemError:
            # Fall through and try to create — the create call will surface
            # any persistent transport errors with a clean message.
            pass

        request_body: dict[str, Any] = {
            "name": name,
            "spaceEmbedders": [
                {"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}
            ],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": chunk_size,
                    "chunkOverlap": chunk_overlap,
                    "separators": list(_DEFAULT_SEPARATORS),
                    "keepStrategy": keep_strategy,
                    "separatorIsRegex": False,
                    "lengthMeasurement": length_measurement,
                }
            },
        }

        try:
            response = self._transport.request_json(
                "POST", "/v1/spaces", request_body
            )
        except GoodMemError as exc:
            return _error_result("create space", exc)

        return {
            "success": True,
            "space_id": response.get("spaceId"),
            "name": response.get("name"),
            "embedder_id": embedder_id,
            "chunking_config": request_body["defaultChunkingConfig"],
            "message": "Space created successfully",
            "reused": False,
        }

    # ------------------------------------------------------------------ #
    # Create Memory
    # ------------------------------------------------------------------ #
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
        """Store a document or text snippet as a memory inside a space.

        Args:
            space_id: Target space ID.
            file_base64: Base64-encoded file content (PDF, DOCX, image,
                etc.). The MIME type is auto-detected from
                ``file_extension``.
            file_extension: File extension used to infer the MIME type
                (e.g. ``"pdf"``, ``"png"``, ``"txt"``).
            text_content: Plain-text body. Ignored when ``file_base64``
                is also supplied.
            source: Where the memory came from — stored as
                ``metadata.source``.
            author: Author or creator of the content — stored as
                ``metadata.author``.
            tags: Comma-separated string or list of tag strings — stored
                as ``metadata.tags`` (an array on the server).
            metadata: Extra key-value metadata merged with the fields above.

        Returns:
            ``{success, memory_id, space_id, status, content_type, message}``
            on success, or ``{success: False, error, details}`` on failure.
        """
        if not file_base64 and not text_content:
            return {
                "success": False,
                "error": "No content provided. Please provide file_base64 or text_content.",
                "details": None,
            }

        request_body: dict[str, Any] = {"spaceId": space_id}

        if file_base64:
            detected = get_mime_type(file_extension) if file_extension else None
            mime_type = detected or "application/octet-stream"
            if mime_type.startswith("text/"):
                try:
                    decoded = base64.b64decode(file_base64).decode("utf-8")
                except (ValueError, UnicodeDecodeError) as exc:
                    return {
                        "success": False,
                        "error": (
                            "Failed to decode file_base64 as UTF-8 text "
                            f"for MIME type {mime_type}: {exc}"
                        ),
                        "details": None,
                    }
                request_body["contentType"] = mime_type
                request_body["originalContent"] = decoded
            else:
                request_body["contentType"] = mime_type
                request_body["originalContentB64"] = file_base64
        else:
            request_body["contentType"] = "text/plain"
            request_body["originalContent"] = text_content

        merged_metadata: dict[str, Any] = {}
        if metadata:
            merged_metadata.update(metadata)
        if source:
            merged_metadata["source"] = source
        if author:
            merged_metadata["author"] = author
        if tags:
            if isinstance(tags, str):
                tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            else:
                tag_list = [t for t in tags if t]
            if tag_list:
                merged_metadata["tags"] = tag_list
        if merged_metadata:
            request_body["metadata"] = merged_metadata

        try:
            response = self._transport.request_json(
                "POST", "/v1/memories", request_body
            )
        except GoodMemError as exc:
            return _error_result("create memory", exc)

        return {
            "success": True,
            "memory_id": response.get("memoryId"),
            "space_id": response.get("spaceId"),
            "status": response.get("processingStatus", "PENDING"),
            "content_type": request_body["contentType"],
            "message": "Memory created successfully",
        }

    # ------------------------------------------------------------------ #
    # Retrieve Memories
    # ------------------------------------------------------------------ #
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    @trace(event_type="retrieval", event_name="goodmem.retrieve_memories")
    def retrieve_memories(
        self,
        query: str,
        space_ids: list[str],
        max_results: int = _DEFAULT_MAX_RESULTS,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
        reranker_id: Optional[str] = None,
        llm_id: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
        llm_temperature: Optional[float] = None,
        chronological_resort: bool = False,
    ) -> dict[str, Any]:
        """Run a semantic similarity search across one or more spaces.

        Args:
            query: Natural-language query.
            space_ids: Space IDs to search across (must be non-empty).
            max_results: Maximum number of results.
            include_memory_definition: Fetch the source-document metadata
                alongside each chunk.
            wait_for_indexing: Retry for up to 10 seconds when no results
                are returned. Use this immediately after creating new
                memories.
            reranker_id: Optional reranker model ID for result ordering.
            llm_id: Optional LLM ID to generate a contextual response.
            relevance_threshold: Minimum score (0–1) for inclusion. Only
                applies when ``reranker_id`` or ``llm_id`` is set.
            llm_temperature: Creativity setting (0–2) for ``llm_id``.
            chronological_resort: Reorder results by creation time.

        Returns:
            ``{success, result_set_id, results, memories, total_results,
            query[, abstract_reply]}``. On failure
            ``{success: False, error, details}``.
        """
        space_keys = [
            {"spaceId": sid.strip()}
            for sid in space_ids
            if sid and sid.strip()
        ]
        if not space_keys:
            return {
                "success": False,
                "error": "At least one space must be selected.",
                "details": None,
            }

        request_body: dict[str, Any] = {
            "message": query,
            "spaceKeys": space_keys,
            "requestedSize": max_results or _DEFAULT_MAX_RESULTS,
            "fetchMemory": bool(include_memory_definition),
        }

        if reranker_id or llm_id:
            config: dict[str, Any] = {}
            if reranker_id:
                config["reranker_id"] = reranker_id
            if llm_id:
                config["llm_id"] = llm_id
            if relevance_threshold is not None:
                config["relevance_threshold"] = relevance_threshold
            if llm_temperature is not None:
                config["llm_temp"] = llm_temperature
            if max_results:
                config["max_results"] = max_results
            if chronological_resort:
                config["chronological_resort"] = True
            request_body["postProcessor"] = {
                "name": "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory",
                "config": config,
            }

        start = time.monotonic()
        last_result: dict[str, Any] = {}

        try:
            while True:
                response_text = self._transport.request_text(
                    "POST", "/v1/memories:retrieve", request_body
                )
                last_result = _parse_retrieve_response(response_text, query)

                if last_result["results"] or not wait_for_indexing:
                    return last_result

                if time.monotonic() - start >= _RETRIEVE_MAX_WAIT_SECONDS:
                    last_result["message"] = (
                        "No results found after waiting for indexing. "
                        "Memories may still be processing."
                    )
                    return last_result

                time.sleep(_RETRIEVE_POLL_INTERVAL_SECONDS)
        except GoodMemError as exc:
            return _error_result("retrieve memories", exc)

    # ------------------------------------------------------------------ #
    # Get Memory
    # ------------------------------------------------------------------ #
    @trace(event_type="tool", event_name="goodmem.get_memory")
    def get_memory(
        self,
        memory_id: str,
        include_content: bool = True,
    ) -> dict[str, Any]:
        """Fetch a memory record by ID.

        Args:
            memory_id: UUID returned from :meth:`create_memory`.
            include_content: When ``True``, also fetch the original document
                content from ``/v1/memories/{id}/content``.

        Returns:
            ``{success, memory[, content][, content_error]}`` on success,
            or ``{success: False, error, details}`` on failure.
        """
        try:
            memory = self._transport.request_json(
                "GET", f"/v1/memories/{memory_id}"
            )
        except GoodMemError as exc:
            return _error_result("get memory", exc)

        result: dict[str, Any] = {"success": True, "memory": memory}
        if include_content:
            try:
                result["content"] = self._transport.request_json(
                    "GET", f"/v1/memories/{memory_id}/content"
                )
            except GoodMemError as exc:
                result["content_error"] = (
                    f"Failed to fetch content: {exc}"
                )
        return result

    # ------------------------------------------------------------------ #
    # Delete Memory
    # ------------------------------------------------------------------ #
    @trace(event_type="tool", event_name="goodmem.delete_memory")
    def delete_memory(self, memory_id: str) -> dict[str, Any]:
        """Permanently delete a memory and its embeddings.

        Args:
            memory_id: UUID of the memory to delete.
        """
        try:
            self._transport.request_json(
                "DELETE", f"/v1/memories/{memory_id}"
            )
        except GoodMemError as exc:
            return _error_result("delete memory", exc)
        return {
            "success": True,
            "memory_id": memory_id,
            "message": "Memory deleted successfully",
        }

    # ------------------------------------------------------------------ #
    # Get Space
    # ------------------------------------------------------------------ #
    @trace(event_type="tool", event_name="goodmem.get_space")
    def get_space(self, space_id: str) -> dict[str, Any]:
        """Fetch a single space by ID.

        Args:
            space_id: UUID of the space to fetch.
        """
        try:
            space = self._transport.request_json(
                "GET", f"/v1/spaces/{space_id}"
            )
        except GoodMemError as exc:
            return _error_result("get space", exc)
        return {"success": True, "space": space}

    # ------------------------------------------------------------------ #
    # Update Space
    # ------------------------------------------------------------------ #
    @trace(event_type="tool", event_name="goodmem.update_space")
    def update_space(
        self,
        space_id: str,
        name: Optional[str] = None,
        public_read: Optional[bool] = None,
        replace_labels: Optional[dict[str, str]] = None,
        merge_labels: Optional[dict[str, str]] = None,
        default_chunking_config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Update mutable fields on a space.

        Only the fields you pass are sent — everything else is left
        untouched. The GoodMem server accepts these fields:
        ``name``, ``publicRead``, ``replaceLabels``, ``mergeLabels``,
        ``defaultChunkingConfig``.

        Args:
            space_id: UUID of the space to update.
            name: New space name.
            public_read: Whether the space is readable by all callers.
            replace_labels: Replaces the existing labels map entirely.
            merge_labels: Merges the supplied labels into the existing map.
            default_chunking_config: Replacement chunking configuration.
        """
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if public_read is not None:
            body["publicRead"] = public_read
        if replace_labels is not None:
            body["replaceLabels"] = replace_labels
        if merge_labels is not None:
            body["mergeLabels"] = merge_labels
        if default_chunking_config is not None:
            body["defaultChunkingConfig"] = default_chunking_config
        if not body:
            return {
                "success": False,
                "error": (
                    "update_space requires at least one field to update "
                    "(name, public_read, replace_labels, merge_labels, "
                    "default_chunking_config)."
                ),
                "details": None,
            }
        try:
            updated = self._transport.request_json(
                "PUT", f"/v1/spaces/{space_id}", body
            )
        except GoodMemError as exc:
            return _error_result("update space", exc)
        return {
            "success": True,
            "space": updated,
            "space_id": updated.get("spaceId") or space_id,
            "message": "Space updated successfully",
        }

    # ------------------------------------------------------------------ #
    # Delete Space
    # ------------------------------------------------------------------ #
    @trace(event_type="tool", event_name="goodmem.delete_space")
    def delete_space(self, space_id: str) -> dict[str, Any]:
        """Permanently delete a space and all of its memories.

        Args:
            space_id: UUID of the space to delete.
        """
        try:
            self._transport.request_json(
                "DELETE", f"/v1/spaces/{space_id}"
            )
        except GoodMemError as exc:
            return _error_result("delete space", exc)
        return {
            "success": True,
            "space_id": space_id,
            "message": "Space deleted successfully",
        }

    # ------------------------------------------------------------------ #
    # List Memories
    # ------------------------------------------------------------------ #
    @trace(event_type="tool", event_name="goodmem.list_memories")
    def list_memories(self, space_id: str) -> dict[str, Any]:
        """List the memories stored in a single space.

        Args:
            space_id: UUID of the space to list memories from.
        """
        try:
            body = self._transport.request_json(
                "GET", f"/v1/spaces/{space_id}/memories"
            )
        except GoodMemError as exc:
            return _error_result("list memories", exc)
        memories = (
            body
            if isinstance(body, list)
            else (body or {}).get("memories") or []
        )
        return {
            "success": True,
            "memories": memories,
            "total_memories": len(memories),
            "space_id": space_id,
        }

    # ------------------------------------------------------------------ #
    # List Spaces
    # ------------------------------------------------------------------ #
    @trace(event_type="tool", event_name="goodmem.list_spaces")
    def list_spaces(self) -> dict[str, Any]:
        """List all spaces visible to the current API key."""
        try:
            body = self._transport.request_json("GET", "/v1/spaces")
        except GoodMemError as exc:
            return _error_result("list spaces", exc)
        spaces = body if isinstance(body, list) else (body or {}).get("spaces") or []
        return {
            "success": True,
            "spaces": [
                {
                    "space_id": s.get("spaceId") or s.get("id"),
                    "name": s.get("name") or "Unnamed",
                }
                for s in spaces
            ],
            "total_spaces": len(spaces),
        }

    # ------------------------------------------------------------------ #
    # List Embedders
    # ------------------------------------------------------------------ #
    @trace(event_type="tool", event_name="goodmem.list_embedders")
    def list_embedders(self) -> dict[str, Any]:
        """List all embedders that can be used when creating a space."""
        try:
            body = self._transport.request_json("GET", "/v1/embedders")
        except GoodMemError as exc:
            return _error_result("list embedders", exc)
        embedders = (
            body if isinstance(body, list) else (body or {}).get("embedders") or []
        )
        return {
            "success": True,
            "embedders": [
                {
                    "embedder_id": e.get("embedderId") or e.get("id"),
                    "display_name": e.get("displayName")
                    or e.get("name")
                    or "Unnamed",
                    "model_identifier": e.get("modelIdentifier")
                    or e.get("model")
                    or "unknown",
                }
                for e in embedders
            ],
            "total_embedders": len(embedders),
        }


def _parse_retrieve_response(text: str, query: str) -> dict[str, Any]:
    """Parse the NDJSON / SSE response body returned by ``/v1/memories:retrieve``.

    The server may stream either NDJSON (one JSON object per line) or
    text/event-stream (``data: {...}`` lines with optional ``event:`` lines).
    Both formats are handled here.
    """
    results: list[dict[str, Any]] = []
    memories: list[Any] = []
    result_set_id = ""
    abstract_reply: Any = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line.startswith("event:"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            # Skip non-JSON lines such as SSE comments or stream terminators.
            continue

        if "resultSetBoundary" in item:
            boundary = item["resultSetBoundary"] or {}
            result_set_id = boundary.get("resultSetId", "")
        elif "memoryDefinition" in item:
            memories.append(item["memoryDefinition"])
        elif "abstractReply" in item:
            abstract_reply = item["abstractReply"]
        elif "retrievedItem" in item:
            chunk_envelope = (item["retrievedItem"] or {}).get("chunk") or {}
            inner_chunk = chunk_envelope.get("chunk") or {}
            results.append(
                {
                    "chunk_id": inner_chunk.get("chunkId"),
                    "chunk_text": inner_chunk.get("chunkText"),
                    "memory_id": inner_chunk.get("memoryId"),
                    "relevance_score": chunk_envelope.get("relevanceScore"),
                    "memory_index": chunk_envelope.get("memoryIndex"),
                }
            )

    output: dict[str, Any] = {
        "success": True,
        "result_set_id": result_set_id,
        "results": results,
        "memories": memories,
        "total_results": len(results),
        "query": query,
    }
    if abstract_reply is not None:
        output["abstract_reply"] = abstract_reply
    return output
