"""Integration tests for the standalone honeyhive-goodmem package.

These tests hit a live GoodMem server and exercise every public method on
``GoodMemClient`` (the same 11 operations the agent-framework integration
exposes as tools).

Run with:
    GOODMEM_API_KEY=<key> GOODMEM_BASE_URL=<url> \\
        python -m pytest tests/test_goodmem_integration.py -v -s -m integration
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path

import pytest

# Allow direct import without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from honeyhive_goodmem import GoodMemClient, GoodMemConfig  # noqa: E402

GOODMEM_API_KEY = os.environ.get("GOODMEM_API_KEY", "gm_g5xcse2tjgcznlg45c5le4ti5q")
GOODMEM_BASE_URL = os.environ.get("GOODMEM_BASE_URL", "https://localhost:8080")
RERANKER_ID = os.environ.get("GOODMEM_RERANKER_ID", "019cfda4-7e2f-743c-9edb-e469a97b95c6")
LLM_ID = os.environ.get("GOODMEM_LLM_ID", "019cfd9f-0963-76f9-b069-4cde19a64ba8")
PREFERRED_EMBEDDER_ID = os.environ.get(
    "GOODMEM_EMBEDDER_ID", "019cfd1c-c033-7517-b7de-f73941a0464b"
)
PDF_FILE_PATH = os.environ.get(
    "GOODMEM_PDF_PATH",
    "/home/bashar/Downloads/New Quran.com Search Analysis (Nov 26, 2025)-1.pdf",
)

UNIQUE_SUFFIX = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
SPACE_NAME = f"hh-goodmem-it-{UNIQUE_SUFFIX}"

_state: dict[str, str] = {}


@pytest.fixture(scope="module")
def client():
    c = GoodMemClient(
        GoodMemConfig(
            base_url=GOODMEM_BASE_URL,
            api_key=GOODMEM_API_KEY,
            verify_ssl=False,
        )
    )
    yield c


@pytest.mark.integration
class TestGoodMemClient:
    """Direct ``GoodMemClient`` tests across every public method."""

    def test_01_list_embedders(self, client: GoodMemClient) -> None:
        result = client.list_embedders()
        print(f"\n[list_embedders] count={result.get('total_embedders')}")
        assert result["success"]
        assert result["total_embedders"] > 0
        chosen = next(
            (e for e in result["embedders"] if e.get("embedder_id") == PREFERRED_EMBEDDER_ID),
            None,
        )
        embedder = chosen or result["embedders"][-1]
        _state["embedder_id"] = embedder["embedder_id"]

    def test_02_create_space(self, client: GoodMemClient) -> None:
        result = client.create_space(name=SPACE_NAME, embedder_id=_state["embedder_id"])
        print(f"\n[create_space] {json.dumps(result, indent=2, default=str)}")
        assert result["success"]
        assert result.get("space_id")
        _state["space_id"] = result["space_id"]

    def test_03_list_spaces(self, client: GoodMemClient) -> None:
        result = client.list_spaces()
        ids = [s["space_id"] for s in result["spaces"]]
        print(f"\n[list_spaces] count={result['total_spaces']} includes_created={_state['space_id'] in ids}")
        assert result["success"]
        assert _state["space_id"] in ids

    def test_04_get_space(self, client: GoodMemClient) -> None:
        result = client.get_space(_state["space_id"])
        print(f"\n[get_space] name={result['space'].get('name')}")
        assert result["success"]
        assert result["space"]["spaceId"] == _state["space_id"]

    def test_05_update_space(self, client: GoodMemClient) -> None:
        new_name = SPACE_NAME + "-updated"
        result = client.update_space(
            _state["space_id"],
            name=new_name,
            replace_labels={"created_by": "pytest"},
        )
        print(f"\n[update_space] {json.dumps(result, indent=2, default=str)}")
        assert result["success"]
        again = client.get_space(_state["space_id"])
        assert again["space"].get("name") == new_name
        _state["space_name"] = new_name

    def test_06_create_memory_text(self, client: GoodMemClient) -> None:
        result = client.create_memory(
            space_id=_state["space_id"],
            text_content=(
                "HoneyHive is an LLM observability platform built on OpenTelemetry. "
                "It captures spans for prompts, tools, retrieval, and evaluation."
            ),
            source="pytest",
            tags="honeyhive,docs",
        )
        print(f"\n[create_memory_text] {json.dumps(result, indent=2)}")
        assert result["success"]
        assert result.get("memory_id")
        _state["text_memory_id"] = result["memory_id"]

    def test_07_create_memory_pdf(self, client: GoodMemClient) -> None:
        if not os.path.isfile(PDF_FILE_PATH):
            pytest.skip(f"PDF not found at {PDF_FILE_PATH}")
        pdf_bytes = Path(PDF_FILE_PATH).read_bytes()
        result = client.create_memory(
            space_id=_state["space_id"],
            file_base64=base64.b64encode(pdf_bytes).decode("ascii"),
            file_extension="pdf",
            source=Path(PDF_FILE_PATH).name,
            tags="pdf,example",
        )
        print(f"\n[create_memory_pdf] {json.dumps(result, indent=2)}")
        assert result["success"]
        assert result.get("content_type") == "application/pdf"
        _state["pdf_memory_id"] = result["memory_id"]

    def test_08_list_memories(self, client: GoodMemClient) -> None:
        result = client.list_memories(_state["space_id"])
        print(f"\n[list_memories] count={result['total_memories']}")
        assert result["success"]
        ids = [m.get("memoryId") for m in result["memories"]]
        assert _state["text_memory_id"] in ids

    def test_09_retrieve_memories(self, client: GoodMemClient) -> None:
        result = client.retrieve_memories(
            query="HoneyHive observability for AI agents",
            space_ids=[_state["space_id"]],
            max_results=5,
            wait_for_indexing=True,
        )
        print(f"\n[retrieve_memories] {result.get('total_results')} chunks")
        assert result["success"]
        assert result["total_results"] > 0

    def test_10_retrieve_with_reranker_and_llm(self, client: GoodMemClient) -> None:
        """Exercise the post-processor parameters (reranker + LLM)."""
        result = client.retrieve_memories(
            query="What does HoneyHive do?",
            space_ids=[_state["space_id"]],
            max_results=3,
            wait_for_indexing=True,
            reranker_id=RERANKER_ID,
            llm_id=LLM_ID,
            relevance_threshold=0.0,
            llm_temperature=0.3,
            chronological_resort=False,
        )
        print(
            f"\n[retrieve_with_reranker_and_llm] "
            f"abstract={bool(result.get('abstract_reply'))}, "
            f"results={result.get('total_results')}"
        )
        assert result["success"]

    def test_11_get_memory(self, client: GoodMemClient) -> None:
        result = client.get_memory(_state["text_memory_id"], include_content=True)
        print(f"\n[get_memory] status={result.get('memory', {}).get('processingStatus')}")
        assert result["success"]
        assert result["memory"]["memoryId"] == _state["text_memory_id"]

    def test_12_delete_memory(self, client: GoodMemClient) -> None:
        result = client.delete_memory(_state["text_memory_id"])
        print(f"\n[delete_memory] {json.dumps(result, indent=2)}")
        assert result["success"]
        if _state.get("pdf_memory_id"):
            r2 = client.delete_memory(_state["pdf_memory_id"])
            assert r2["success"]

    def test_13_delete_space(self, client: GoodMemClient) -> None:
        result = client.delete_space(_state["space_id"])
        print(f"\n[delete_space] {json.dumps(result, indent=2)}")
        assert result["success"]
