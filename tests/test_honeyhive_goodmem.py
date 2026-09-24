"""Offline tests for honeyhive-goodmem.

These drive the *real* GoodMem SDK over an ``httpx`` mock transport, fed with
NDJSON and JSON captured from a live GoodMem server (v1.0.320).

0.1.0 shipped no offline tests at all: its 13 "tests" hit a live server and
silently fell back to a real API key committed in the file, so they passed on
the author's machine and told nobody anything about the defects.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest

from honeyhive_goodmem import (
    GoodMemClient,
    GoodMemConfig,
    GoodMemError,
    filters,
)
from honeyhive_goodmem._filters import GoodMemFilterError
from honeyhive_goodmem._results import (
    MALFORMED_STREAM_CODE,
    UNKNOWN_CODE,
    classify_status,
    orient_score,
    outcome_from_events,
)

FIXTURES = Path(__file__).parent / "goodmem_fixtures"
BASE = "https://goodmem.test"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def make_client(handler, **kwargs) -> GoodMemClient:
    from goodmem import Goodmem

    sdk = Goodmem(
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url=BASE,
            headers={"X-API-Key": "gm_offline_test_key"},
        ),
    )
    return GoodMemClient(
        GoodMemConfig(base_url=BASE, api_key="gm_offline_test_key"),
        client=sdk,
        **kwargs,
    )


def retrieve_handler(payload: bytes, *, capture: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(":retrieve"):
            if capture is not None:
                capture["body"] = json.loads(request.content)
            return httpx.Response(
                200, content=payload, headers={"content-type": "application/x-ndjson"}
            )
        return httpx.Response(404, json={"message": "unexpected"})

    return handler


class TestNoCommittedCredential:
    def test_no_api_key_anywhere_in_the_tree(self):
        """0.1.0 defaulted GOODMEM_API_KEY to a live key, on the default branch."""
        root = Path(__file__).resolve().parent.parent
        offenders = []
        for path in root.rglob("*"):
            if not path.is_file() or ".git" in path.parts or ".venv" in path.parts:
                continue
            if path.suffix not in {".py", ".toml", ".md", ".yml", ".yaml", ".txt"}:
                continue
            if re.search(r"gm_[a-z0-9]{20,}", path.read_text(errors="ignore")):
                offenders.append(str(path.relative_to(root)))
        assert offenders == [], f"credential-shaped string in {offenders}"

    def test_fixtures_are_real_server_bytes(self):
        stream = fixture("retrieve_ok.ndjson").decode()
        events = [json.loads(x) for x in stream.strip().split("\n") if x.strip()]
        assert any("resultSetBoundary" in e for e in events)
        assert any("retrievedItem" in e for e in events)
        assert re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-", stream, re.I)


class TestRetrievalStatusContract:
    def test_a_degraded_retrieval_is_visible_in_the_traced_payload(self):
        """The whole point of this package: a span must not say success."""
        c = make_client(retrieve_handler(fixture("retrieve_degraded_hits.ndjson")))
        payload = c.retrieve_memories("canary", ["s-1"])
        assert payload["partial"] is True
        assert payload["statuses"]
        assert payload["warning"]
        assert payload["total_results"] > 0, "hits were discarded"

    def test_q4b_degraded_with_no_hits_is_flagged(self):
        c = make_client(retrieve_handler(fixture("retrieve_degraded_empty.ndjson")))
        payload = c.retrieve_memories("nothing", ["s-1"])
        assert payload["total_results"] == 0
        assert payload["partial"] is True and payload["statuses"]

    def test_q1_informational_codes_are_noise(self):
        assert classify_status("FEATURE_DISABLED", "x").informational is True
        assert classify_status("LLM_CAPABILITY_INFERRED", "x").informational is True
        assert classify_status("EMBEDDER_FAILED", "x").informational is False

    def test_q3_unknown_code_becomes_unknown(self):
        payload = (
            json.dumps({"status": {"code": "FUTURE", "message": "new"}}).encode()
            + b"\n"
        )
        c = make_client(retrieve_handler(payload))
        out = c.retrieve_memories("q", ["s-1"])
        assert [s["code"] for s in out["statuses"]] == [UNKNOWN_CODE]
        assert out["partial"] is True

    def test_a_clean_retrieval_is_not_partial(self):
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        out = c.retrieve_memories("canary", ["s-1"])
        assert out["partial"] is False
        assert out["statuses"] == []
        assert "warning" not in out

    def test_a_truncated_stream_keeps_what_arrived(self):
        whole = fixture("retrieve_ok.ndjson")
        c = make_client(retrieve_handler(whole[: int(len(whole) * 0.6)]))
        out = c.retrieve_memories("canary", ["s-1"])
        assert out["partial"] is True
        assert MALFORMED_STREAM_CODE in {s["code"] for s in out["statuses"]}


class TestTracedPayloadShape:
    def test_results_carry_ids_score_and_metadata(self):
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        hit = c.retrieve_memories("canary", ["s-1"])["results"][0]
        for key in (
            "chunk_id",
            "memory_id",
            "space_id",
            "score",
            "raw_score",
            "score_kind",
            "metadata",
        ):
            assert key in hit, f"{key} missing from the traced payload"

    def test_scores_are_oriented_with_the_raw_value_kept(self):
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        hit = c.retrieve_memories("canary", ["s-1"])["results"][0]
        assert hit["raw_score"] < 0 and hit["score"] > 0
        assert hit["score"] == pytest.approx(-hit["raw_score"])
        assert hit["score_kind"] == "vector"

    def test_vector_and_reranker_scores_are_oriented_differently(self):
        assert orient_score(-0.51, reranked=False) == pytest.approx(0.51)
        assert orient_score(0.93, reranked=True) == pytest.approx(0.93)
        assert orient_score(-0.14, reranked=True) == pytest.approx(-0.14)

    def test_the_payload_is_json_serialisable(self):
        c = make_client(retrieve_handler(fixture("retrieve_degraded_hits.ndjson")))
        json.dumps(c.retrieve_memories("canary", ["s-1"]))

    def test_no_relevance_threshold_is_sent_by_default(self):
        capture: dict = {}
        c = make_client(
            retrieve_handler(fixture("retrieve_ok.ndjson"), capture=capture)
        )
        c.retrieve_memories("canary", ["s-1"])
        assert "relevanceThreshold" not in json.dumps(capture["body"])


class TestPublicReadRemoved:
    def test_update_space_has_no_public_read(self):
        import inspect

        assert (
            "public_read"
            not in inspect.signature(GoodMemClient.update_space).parameters
        )

    def test_no_public_read_in_any_shipped_code_path(self):
        import ast

        import honeyhive_goodmem

        offenders = []
        for path in Path(honeyhive_goodmem.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    node.value = ""
            src = ast.unparse(tree)
            if "publicRead" in src or "public_read" in src:
                offenders.append(path.name)
        assert offenders == []


class TestSpaces:
    def _spaces(self, spaces):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path == "/v1/spaces":
                return httpx.Response(200, json={"spaces": spaces})
            return httpx.Response(404, json={"message": "unexpected"})

        return handler

    def _space(self, space_id, name, embedders):
        import copy

        t = copy.deepcopy(json.loads(fixture("spaces_page1.json"))["spaces"][0])
        t["spaceId"], t["name"] = space_id, name
        et = t["spaceEmbedders"][0]
        t["spaceEmbedders"] = []
        for e in embedders:
            clone = copy.deepcopy(et)
            clone["embedderId"], clone["spaceId"] = e, space_id
            t["spaceEmbedders"].append(clone)
        return t

    def test_reuse_requires_a_matching_embedder(self):
        c = make_client(self._spaces([self._space("s-1", "notes", ["emb-a"])]))
        with pytest.raises(GoodMemError) as err:
            c.create_space("notes", "emb-b")
        assert "emb-a" in str(err.value) and "emb-b" in str(err.value)

    def test_reuse_with_a_match_succeeds(self):
        c = make_client(self._spaces([self._space("s-1", "notes", ["emb-a"])]))
        out = c.create_space("notes", "emb-a")
        assert out["reused"] is True and out["space_id"] == "s-1"

    def test_an_ambiguous_name_is_an_error(self):
        c = make_client(
            self._spaces(
                [self._space("s-1", "n", ["e"]), self._space("s-2", "n", ["e"])]
            )
        )
        with pytest.raises(GoodMemError, match="refusing to guess"):
            c.create_space("n", "e")

    def test_listing_follows_pagination(self):
        p1 = json.loads(fixture("spaces_page1.json"))
        p2 = json.loads(fixture("spaces_page2.json"))
        p2.pop("nextToken", None)
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, json=p1 if len(calls) == 1 else p2)

        out = make_client(handler).list_spaces()
        assert len(calls) == 2, "the second page was never requested"
        assert out["total_results"] == len(p1["spaces"]) + len(p2["spaces"])

    def test_the_servers_message_reaches_the_caller(self):
        body = fixture("error_400.json")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400, content=body, headers={"content-type": "application/json"}
            )

        with pytest.raises(GoodMemError) as err:
            make_client(handler).list_spaces()
        assert "Invalid embedder ID format" in str(err.value)


class TestContentAndSecrets:
    def _handler(self, content: bytes, content_type: str, status: int = 200):
        memory = json.loads(fixture("memory_get.json"))
        memory["contentType"] = content_type

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/content"):
                return httpx.Response(
                    status, content=content, headers={"content-type": content_type}
                )
            return httpx.Response(200, json=memory)

        return handler

    def test_text_content_comes_back_as_text(self):
        out = make_client(self._handler(b"hello", "text/plain")).get_memory(
            "m-1", include_content=True
        )
        assert out["content"] == "hello" and out["content_encoding"] == "text"

    def test_binary_content_is_base64_and_serialisable(self):
        import base64

        pdf = b"%PDF-1.4\x00\xff"
        out = make_client(self._handler(pdf, "application/pdf")).get_memory(
            "m-1", include_content=True
        )
        json.dumps(out)
        assert base64.b64decode(out["content"]) == pdf

    def test_a_failed_content_fetch_raises(self):
        c = make_client(self._handler(b'{"m":"gone"}', "application/json", 404))
        with pytest.raises(GoodMemError):
            c.get_memory("m-1", include_content=True)

    def test_the_api_key_is_not_in_repr(self):
        assert "gm_offline_test_key" not in repr(make_client(retrieve_handler(b"")))

    def test_the_api_key_is_not_a_public_attribute(self):
        c = make_client(retrieve_handler(b""))
        public = {
            v
            for k, v in vars(c).items()
            if not k.startswith("_") and isinstance(v, str)
        }
        assert "gm_offline_test_key" not in public

    def test_an_injected_client_is_not_closed(self):
        c = make_client(retrieve_handler(b""))
        c.close()
        assert c._owns_client is False


class TestFilters:
    def test_apostrophes_are_backslash_escaped(self):
        assert filters.equals("n", "o'brien").endswith(r"'o\'brien'")

    def test_control_characters_are_refused(self):
        with pytest.raises(GoodMemFilterError, match="control characters"):
            filters.equals("f", "a\nb")

    def test_booleans_cast_as_boolean(self):
        assert filters.equals("a", True) == "CAST(val('$.a') AS BOOLEAN) = true"

    def test_unsafe_field_names_are_refused(self):
        with pytest.raises(GoodMemFilterError, match="field name"):
            filters.equals("a' OR '1", "x")

    def test_the_filter_reaches_the_request(self):
        capture: dict = {}
        c = make_client(
            retrieve_handler(fixture("retrieve_ok.ndjson"), capture=capture)
        )
        c.retrieve_memories("q", ["s-1"], metadata_filter={"tenant": "acme"})
        assert (
            capture["body"]["spaceKeys"][0]["filter"]
            == "CAST(val('$.tenant') AS TEXT) = 'acme'"
        )


class TestJoin:
    def _event(self, kind):
        import copy

        for line in fixture("retrieve_ok.ndjson").decode().strip().split("\n"):
            e = json.loads(line)
            if kind in e:
                return copy.deepcopy(e)
        raise AssertionError(f"no {kind} in the fixture")

    def _chunk(self, chunk_id, memory_id, score):
        e = self._event("retrievedItem")
        ref = e["retrievedItem"]["chunk"]
        ref["relevanceScore"] = score
        ref["chunk"]["chunkId"] = chunk_id
        ref["chunk"]["memoryId"] = memory_id
        return e

    def _definition(self, memory_id, metadata):
        e = self._event("memoryDefinition")
        e["memoryDefinition"]["memoryId"] = memory_id
        e["memoryDefinition"]["metadata"] = metadata
        return e

    def _models(self, events):
        from goodmem.models import RetrieveMemoryEvent

        return [RetrieveMemoryEvent.model_validate(e) for e in events]

    def test_join_is_by_uuid_not_arrival_order(self):
        out = outcome_from_events(
            self._models(
                [
                    self._chunk("c1", "mem-A", -0.2),
                    self._chunk("c2", "mem-B", -0.4),
                    self._definition("mem-B", {"tag": "B"}),
                    self._definition("mem-A", {"tag": "A"}),
                ]
            )
        )
        by_id = {h.chunk_id: h for h in out.hits}
        assert by_id["c1"].metadata == {"tag": "A"}
        assert by_id["c2"].metadata == {"tag": "B"}

    def test_duplicates_collapse_but_distinct_chunks_survive(self):
        dup = outcome_from_events(
            self._models([self._chunk("c1", "m", -0.2), self._chunk("c1", "m", -0.2)])
        )
        assert len(dup.hits) == 1
        two = outcome_from_events(
            self._models([self._chunk("c1", "m", -0.2), self._chunk("c2", "m", -0.3)])
        )
        assert len(two.hits) == 2
