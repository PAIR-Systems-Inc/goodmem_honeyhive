"""Live tests for honeyhive-goodmem, against a running GoodMem server.

These skip entirely unless GOODMEM_API_KEY and GOODMEM_BASE_URL are set.
0.1.0's suite fell back to a key committed in the file, so it ran whether or
not anyone meant it to; there is no fallback here.
"""

from __future__ import annotations

import base64
import json
import os
import time
import uuid

import pytest

from honeyhive_goodmem import GoodMemClient, GoodMemConfig, GoodMemError

API_KEY = os.environ.get("GOODMEM_API_KEY")
BASE_URL = os.environ.get("GOODMEM_BASE_URL")
VERIFY_SSL = os.environ.get("GOODMEM_VERIFY_SSL", "false").lower() == "true"
FAILING_EMBEDDER = os.environ.get("GOODMEM_TEST_FAILING_EMBEDDER_ID")

pytestmark = pytest.mark.skipif(
    not (API_KEY and BASE_URL),
    reason="GOODMEM_API_KEY and GOODMEM_BASE_URL are not set",
)

RUN = uuid.uuid4().hex[:8]


def _embedder_id(client: GoodMemClient) -> str:
    pinned = os.environ.get("GOODMEM_TEST_EMBEDDER_ID")
    if pinned:
        return pinned
    embedders = client.list_embedders()["embedders"]
    assert embedders, "the server has no embedders configured"
    return embedders[0]["embedder_id"]


@pytest.fixture(scope="module")
def client() -> GoodMemClient:
    c = GoodMemClient(
        GoodMemConfig(base_url=BASE_URL, api_key=API_KEY, verify_ssl=VERIFY_SSL)
    )
    yield c
    c.close()


@pytest.fixture(scope="module")
def space(client: GoodMemClient):
    created = client.create_space(f"hh-live-{RUN}", _embedder_id(client))
    space_id = created["space_id"]
    yield space_id
    client.delete_space(space_id)
    remaining = [s for s in client.list_spaces()["spaces"] if s["space_id"] == space_id]
    assert remaining == [], "the space survived teardown"


@pytest.fixture(scope="module")
def seeded(client: GoodMemClient, space: str):
    canary = f"ORYX-{RUN.upper()}"
    created = client.create_memory(
        space,
        text_content=f"The HoneyHive live canary is {canary}.",
        metadata={"tenant": "acme"},
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        if client.retrieve_memories(canary, [space], max_results=3)["total_results"]:
            break
        time.sleep(2)
    else:
        pytest.fail("the seeded memory never became searchable")
    yield canary, created["memory_id"]
    client.delete_memory(created["memory_id"])


class TestLiveRetrieval:
    def test_an_exact_identifier_round_trips(self, client, space, seeded):
        canary, memory_id = seeded
        out = client.retrieve_memories(canary, [space], max_results=5)
        assert out["partial"] is False
        assert any(canary in r["chunk_text"] for r in out["results"])
        assert out["results"][0]["memory_id"] == memory_id

    def test_the_traced_payload_carries_score_ids_and_metadata(
        self, client, space, seeded
    ):
        hit = client.retrieve_memories(seeded[0], [space], max_results=1)["results"][0]
        assert hit["raw_score"] < 0, "GoodMem vector scores are negative"
        assert hit["score"] > 0, "not flipped to higher-is-better"
        assert hit["score_kind"] == "vector"
        assert hit["metadata"]["tenant"] == "acme"
        assert hit["chunk_id"] and hit["memory_id"] and hit["space_id"]

    def test_the_payload_is_json_serialisable(self, client, space, seeded):
        json.dumps(client.retrieve_memories(seeded[0], [space], max_results=3))

    def test_a_failing_embedder_is_visible_in_the_payload(self, client):
        """0.1.0 recorded this as a successful span with zero results."""
        if not FAILING_EMBEDDER:
            pytest.skip("GOODMEM_TEST_FAILING_EMBEDDER_ID is not set")
        bad = client.create_space(f"hh-live-bad-{RUN}", FAILING_EMBEDDER)
        created = None
        try:
            created = client.create_memory(
                bad["space_id"], text_content="Doomed canary."
            )
            time.sleep(6)
            out = client.retrieve_memories(
                "doomed canary", [bad["space_id"]], max_results=3
            )
            assert out["partial"] is True
            assert out["statuses"], "the server's status was dropped"
            assert out["warning"]
        finally:
            if created:
                client.delete_memory(created["memory_id"])
            client.delete_space(bad["space_id"])

    def test_a_healthy_search_reports_no_status(self, client, space, seeded):
        out = client.retrieve_memories(seeded[0], [space], max_results=3)
        assert out["partial"] is False and out["statuses"] == []

    def test_the_read_path_does_not_poll(self, client):
        empty = client.create_space(f"hh-live-fast-{RUN}", _embedder_id(client))
        try:
            started = time.time()
            out = client.retrieve_memories(
                "nothing at all", [empty["space_id"]], max_results=3
            )
            elapsed = time.time() - started
        finally:
            client.delete_space(empty["space_id"])
        assert out["total_results"] == 0
        assert elapsed < 3.0, f"an empty search took {elapsed:.1f}s"


class TestLiveFilters:
    def test_a_matching_filter_finds_it(self, client, space, seeded):
        out = client.retrieve_memories(
            seeded[0], [space], max_results=5, metadata_filter={"tenant": "acme"}
        )
        assert out["total_results"] > 0

    def test_an_injection_payload_matches_nothing(self, client, space, seeded):
        out = client.retrieve_memories(
            seeded[0],
            [space],
            max_results=5,
            metadata_filter={"tenant": "x' OR '1'='1"},
        )
        assert out["total_results"] == 0, "filter injection succeeded"

    def test_an_apostrophe_is_accepted(self, client, space):
        out = client.retrieve_memories(
            "anything", [space], max_results=1, metadata_filter={"tenant": "o'brien"}
        )
        assert out["total_results"] == 0


class TestLiveSpaces:
    def test_rename_works_without_public_read(self, client, space):
        renamed = f"hh-live-{RUN}-renamed"
        assert client.update_space(space, name=renamed)["success"] is True
        assert client.get_space(space)["name"] == renamed
        client.update_space(space, name=f"hh-live-{RUN}")

    def test_reuse_with_another_embedder_is_refused(self, client, space):
        others = [
            e["embedder_id"]
            for e in client.list_embedders()["embedders"]
            if e["embedder_id"] != _embedder_id(client)
        ]
        if not others:
            pytest.skip("need two embedders")
        with pytest.raises(GoodMemError, match="cannot be changed"):
            client.create_space(f"hh-live-{RUN}", others[0])

    def test_a_rejected_create_carries_the_servers_message(self, client):
        with pytest.raises(GoodMemError) as err:
            client.create_space(f"hh-live-bad-{RUN}", "not-a-uuid")
        assert "embedder" in str(err.value).lower()

    def test_listing_is_paginated_and_unique(self, client):
        spaces = client.list_spaces()["spaces"]
        assert len({s["space_id"] for s in spaces}) == len(spaces)
        assert any(s["name"].startswith(f"hh-live-{RUN}") for s in spaces)


class TestLiveContent:
    def test_text_content_round_trips(self, client, seeded):
        out = client.get_memory(seeded[1], include_content=True)
        assert out["content_encoding"] == "text" and seeded[0] in out["content"]

    def test_binary_content_round_trips_as_base64(self, client, space):
        pdf = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"
        )
        created = client.create_memory(
            space,
            file_base64=base64.b64encode(pdf).decode("ascii"),
            file_extension="pdf",
        )
        try:
            time.sleep(3)
            out = client.get_memory(created["memory_id"], include_content=True)
            json.dumps(out)
            assert out["content_encoding"] == "base64"
            assert base64.b64decode(out["content"]) == pdf
        finally:
            client.delete_memory(created["memory_id"])

    def test_listing_memories_finds_the_seeded_one(self, client, space, seeded):
        ids = [m["memory_id"] for m in client.list_memories(space)["memories"]]
        assert seeded[1] in ids
