"""Offline tests for honeyhive-goodmem.

These drive the *real* GoodMem SDK over an ``httpx`` mock transport, fed with
NDJSON and JSON captured from a live GoodMem server (v1.0.320). The id tests
go further and send real HTTP to a local server that records every request.

0.1.0 shipped no offline tests at all: its 13 "tests" hit a live server and
silently fell back to a real API key committed in the file, so they passed on
the author's machine and told nobody anything about the defects.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

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

# Real ids from the captured fixtures. Every GoodMem id is a UUID, and the
# client refuses anything else before a request is made.
SPACE_ID = "01a0d44b-746f-775b-b91e-bc73d4058e27"
OTHER_SPACE_ID = "01a0d44b-96ae-7081-bc16-5644e701222a"
MEMORY_ID = "01a0d44b-748d-72eb-b54e-c3ea2d956927"
EMB_A = "019cfd1c-c033-7517-b7de-f73941a0464b"
EMB_B = "019e3d24-0763-70f5-9786-da6b30b90d2f"


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
        payload = c.retrieve_memories("canary", [SPACE_ID])
        assert payload["partial"] is True
        assert payload["statuses"]
        assert payload["warning"]
        assert payload["total_results"] > 0, "hits were discarded"

    def test_q4b_degraded_with_no_hits_is_flagged(self):
        c = make_client(retrieve_handler(fixture("retrieve_degraded_empty.ndjson")))
        payload = c.retrieve_memories("nothing", [SPACE_ID])
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
        out = c.retrieve_memories("q", [SPACE_ID])
        assert [s["code"] for s in out["statuses"]] == [UNKNOWN_CODE]
        assert out["partial"] is True

    def test_a_clean_retrieval_is_not_partial(self):
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        out = c.retrieve_memories("canary", [SPACE_ID])
        assert out["partial"] is False
        assert out["statuses"] == []
        assert "warning" not in out

    def test_a_truncated_stream_keeps_what_arrived(self):
        whole = fixture("retrieve_ok.ndjson")
        c = make_client(retrieve_handler(whole[: int(len(whole) * 0.6)]))
        out = c.retrieve_memories("canary", [SPACE_ID])
        assert out["partial"] is True
        assert MALFORMED_STREAM_CODE in {s["code"] for s in out["statuses"]}


class TestTracedPayloadShape:
    def test_results_carry_ids_score_and_metadata(self):
        c = make_client(retrieve_handler(fixture("retrieve_ok.ndjson")))
        hit = c.retrieve_memories("canary", [SPACE_ID])["results"][0]
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
        hit = c.retrieve_memories("canary", [SPACE_ID])["results"][0]
        assert hit["raw_score"] < 0 and hit["score"] > 0
        assert hit["score"] == pytest.approx(-hit["raw_score"])
        assert hit["score_kind"] == "vector"

    def test_vector_and_reranker_scores_are_oriented_differently(self):
        assert orient_score(-0.51, reranked=False) == pytest.approx(0.51)
        assert orient_score(0.93, reranked=True) == pytest.approx(0.93)
        assert orient_score(-0.14, reranked=True) == pytest.approx(-0.14)

    def test_the_payload_is_json_serialisable(self):
        c = make_client(retrieve_handler(fixture("retrieve_degraded_hits.ndjson")))
        json.dumps(c.retrieve_memories("canary", [SPACE_ID]))

    def test_no_relevance_threshold_is_sent_by_default(self):
        capture: dict = {}
        c = make_client(
            retrieve_handler(fixture("retrieve_ok.ndjson"), capture=capture)
        )
        c.retrieve_memories("canary", [SPACE_ID])
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
        c = make_client(self._spaces([self._space(SPACE_ID, "notes", [EMB_A])]))
        with pytest.raises(GoodMemError) as err:
            c.create_space("notes", EMB_B)
        assert EMB_A in str(err.value) and EMB_B in str(err.value)

    def test_reuse_with_a_match_succeeds(self):
        c = make_client(self._spaces([self._space(SPACE_ID, "notes", [EMB_A])]))
        out = c.create_space("notes", EMB_A)
        assert out["reused"] is True and out["space_id"] == SPACE_ID

    def test_an_ambiguous_name_is_an_error(self):
        c = make_client(
            self._spaces(
                [
                    self._space(SPACE_ID, "n", [EMB_A]),
                    self._space(OTHER_SPACE_ID, "n", [EMB_A]),
                ]
            )
        )
        with pytest.raises(GoodMemError, match="refusing to guess"):
            c.create_space("n", EMB_A)

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
            MEMORY_ID, include_content=True
        )
        assert out["content"] == "hello" and out["content_encoding"] == "text"

    def test_binary_content_is_base64_and_serialisable(self):
        import base64

        pdf = b"%PDF-1.4\x00\xff"
        out = make_client(self._handler(pdf, "application/pdf")).get_memory(
            MEMORY_ID, include_content=True
        )
        json.dumps(out)
        assert base64.b64decode(out["content"]) == pdf

    def test_a_failed_content_fetch_raises(self):
        c = make_client(self._handler(b'{"m":"gone"}', "application/json", 404))
        with pytest.raises(GoodMemError):
            c.get_memory(MEMORY_ID, include_content=True)

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
        c.retrieve_memories("q", [SPACE_ID], metadata_filter={"tenant": "acme"})
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


# ---------------------------------------------------------------------------
# Ids that reach a URL path
# ---------------------------------------------------------------------------
#
# The GoodMem SDK builds paths as f"/v1/memories/{id}" with the id raw, and
# httpx resolves dot segments before sending. On 0.2.0,
# delete_memory("../spaces/<id>") therefore sent DELETE /v1/spaces/<id> -- it
# deleted a whole space and reported success. These tests drive the real SDK
# and httpx over a real socket to a local server that records every request.

VICTIM = "01a0d44b-96ae-7081-bc16-5644e701222a"

NON_UUID_IDS = [
    f"../spaces/{VICTIM}",
    f"a/../../spaces/{VICTIM}",
    f"%2e%2e/spaces/{VICTIM}",
    f"..%2Fspaces%2F{VICTIM}",
    f"{VICTIM}/../../spaces/{VICTIM}",
    "",
    f" {VICTIM}",
    f"{VICTIM}?x=1",
    f"{VICTIM}#frag",
    # A regex anchored with `$` accepts a trailing newline; fullmatch does not.
    f"{VICTIM}\n",
    VICTIM.replace("-", ""),
    None,
]


def _space_json(space_id: str) -> dict:
    space = json.loads(fixture("spaces_page1.json"))["spaces"][0]
    space["spaceId"] = space_id
    return space


def _reply(method: str, path: str) -> tuple[int, str, bytes]:
    """A plausible GoodMem answer for every route the client uses."""
    as_json = "application/json"
    if method == "DELETE":
        return 204, as_json, b""
    if method == "POST" and path == "/v1/memories:retrieve":
        return 200, "application/x-ndjson", fixture("retrieve_ok.ndjson")
    if method == "POST" and path == "/v1/memories":
        return 200, as_json, fixture("memory_get.json")
    if method == "POST" and path == "/v1/spaces":
        return 200, as_json, json.dumps(_space_json(SPACE_ID)).encode()
    if method == "GET" and path == "/v1/spaces":
        return 200, as_json, b'{"spaces": []}'
    if method == "GET" and path.endswith("/memories"):
        return 200, as_json, b'{"memories": []}'
    if method == "GET" and path.endswith("/content"):
        return 200, "text/plain", b"hello"
    if method in {"GET", "PUT"} and path.startswith("/v1/spaces/"):
        return 200, as_json, json.dumps(_space_json(path.split("/")[3])).encode()
    if method == "GET" and path.startswith("/v1/memories/"):
        return 200, as_json, fixture("memory_get.json")
    return 404, as_json, b'{"message": "unexpected"}'


class _Recorder(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _handle(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.requests.append((self.command, self.path, body))  # type: ignore[attr-defined]
        status, content_type, payload = _reply(self.command, urlsplit(self.path).path)
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle

    def log_message(self, *args) -> None:
        pass


@pytest.fixture(scope="module")
def recording_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def recorded(recording_server, monkeypatch):
    """A client built the ordinary way, pointed at the recording server."""
    recording_server.requests.clear()
    host, port = recording_server.server_address
    # Keep a developer's HTTP(S)_PROXY from routing these past the recorder.
    monkeypatch.setenv("NO_PROXY", host)
    monkeypatch.setenv("no_proxy", host)
    client = GoodMemClient(
        GoodMemConfig(base_url=f"http://{host}:{port}", api_key="gm_offline_test_key")
    )
    yield client, recording_server.requests
    client.close()


# Every client method that takes an id, with the call that passes the id under
# test and the requests a valid id must produce. Ids in a path are the
# traversal risk; ids in a request body go through the same check anyway.
#   label: (method, id parameter, call, [(verb, path)] for a valid id)
ENTRY_POINTS = {
    "get_space": (
        "get_space",
        "space_id",
        lambda c, v: c.get_space(v),
        [("GET", "/v1/spaces/{u}")],
    ),
    "update_space": (
        "update_space",
        "space_id",
        lambda c, v: c.update_space(v, name="renamed"),
        [("PUT", "/v1/spaces/{u}")],
    ),
    "delete_space": (
        "delete_space",
        "space_id",
        lambda c, v: c.delete_space(v),
        [("DELETE", "/v1/spaces/{u}")],
    ),
    "list_memories": (
        "list_memories",
        "space_id",
        lambda c, v: c.list_memories(v),
        [("GET", "/v1/spaces/{u}/memories")],
    ),
    "get_memory": (
        "get_memory",
        "memory_id",
        lambda c, v: c.get_memory(v),
        [("GET", "/v1/memories/{u}")],
    ),
    "get_memory+content": (
        "get_memory",
        "memory_id",
        lambda c, v: c.get_memory(v, include_content=True),
        [("GET", "/v1/memories/{u}"), ("GET", "/v1/memories/{u}/content")],
    ),
    "delete_memory": (
        "delete_memory",
        "memory_id",
        lambda c, v: c.delete_memory(v),
        [("DELETE", "/v1/memories/{u}")],
    ),
    "create_memory": (
        "create_memory",
        "space_id",
        lambda c, v: c.create_memory(v, text_content="x"),
        [("POST", "/v1/memories")],
    ),
    "create_space": (
        "create_space",
        "embedder_id",
        lambda c, v: c.create_space("notes", v),
        [("GET", "/v1/spaces"), ("POST", "/v1/spaces")],
    ),
    "retrieve_memories[list]": (
        "retrieve_memories",
        "space_ids",
        lambda c, v: c.retrieve_memories("q", [SPACE_ID, v]),
        [("POST", "/v1/memories:retrieve")],
    ),
    "retrieve_memories[str]": (
        "retrieve_memories",
        "space_ids",
        lambda c, v: c.retrieve_memories("q", v),
        [("POST", "/v1/memories:retrieve")],
    ),
    "retrieve_memories[reranker]": (
        "retrieve_memories",
        "reranker_id",
        lambda c, v: c.retrieve_memories("q", [SPACE_ID], reranker_id=v),
        [("POST", "/v1/memories:retrieve")],
    ),
    "retrieve[list]": (
        "retrieve",
        "space_ids",
        lambda c, v: c.retrieve("q", [SPACE_ID, v]),
        [("POST", "/v1/memories:retrieve")],
    ),
    "retrieve[reranker]": (
        "retrieve",
        "reranker_id",
        lambda c, v: c.retrieve("q", [SPACE_ID], reranker_id=v),
        [("POST", "/v1/memories:retrieve")],
    ),
}

REFUSAL_CASES = [
    pytest.param(label, bad, id=f"{label}-{bad!r}")
    for label, (_, param, _, _) in ENTRY_POINTS.items()
    for bad in NON_UUID_IDS
    # reranker_id is optional: None means "no reranker", not a bad id.
    if not (bad is None and param == "reranker_id")
]


def _call(fn):
    try:
        return fn(), None
    except Exception as exc:
        return None, exc


class TestIdsNeverReachAPathUnchecked:
    @pytest.mark.parametrize(("label", "bad"), REFUSAL_CASES)
    def test_a_non_uuid_id_is_refused_before_any_request(self, recorded, label, bad):
        client, requests = recorded
        _, param, call, _ = ENTRY_POINTS[label]
        result, exc = _call(lambda: call(client, bad))
        sent = [f"{verb} {path}" for verb, path, _ in requests]
        assert sent == [], (
            f"{label} with {bad!r} reached the server as {sent}; "
            f"it returned {result!r}"
        )
        assert isinstance(exc, GoodMemError), f"not refused: {exc!r}"
        assert param in str(exc) and "must be a UUID" in str(exc)
        assert result is None, "a refused call must not report success"

    @pytest.mark.parametrize("given", [VICTIM, VICTIM.upper()], ids=["lower", "upper"])
    @pytest.mark.parametrize("label", list(ENTRY_POINTS))
    def test_a_valid_uuid_reaches_exactly_the_intended_path(
        self, recorded, label, given
    ):
        client, requests = recorded
        _, param, call, expected = ENTRY_POINTS[label]
        call(client, given)
        sent = [(verb, urlsplit(path).path) for verb, path, _ in requests]
        assert sent == [(verb, path.format(u=VICTIM)) for verb, path in expected]
        if expected[-1][0] == "POST":
            # A body-carried id is sent in its canonical, lower-case form.
            body = requests[-1][2].decode()
            assert VICTIM in body and VICTIM.upper() not in body

    def test_a_uuid_object_is_accepted(self, recorded):
        client, requests = recorded
        client.delete_memory(uuid.UUID(VICTIM))  # type: ignore[arg-type]
        assert [(verb, path) for verb, path, _ in requests] == [
            ("DELETE", f"/v1/memories/{VICTIM}")
        ]

    def test_every_id_parameter_is_covered(self):
        """A new id-taking method must be added to ENTRY_POINTS above."""
        covered = {(method, param) for method, param, _, _ in ENTRY_POINTS.values()}
        found = set()
        for name, fn in inspect.getmembers(GoodMemClient, inspect.isfunction):
            if name.startswith("_"):
                continue
            for param in inspect.signature(fn).parameters:
                if param.endswith(("_id", "_ids")):
                    found.add((name, param))
        assert found == covered

    @pytest.mark.parametrize(
        "bad",
        [f"{{{VICTIM}}}", f"urn:uuid:{VICTIM}", VICTIM.encode(), 123, VICTIM[:-1]],
        ids=repr,
    )
    def test_the_validator_accepts_only_the_canonical_form(self, bad):
        from honeyhive_goodmem._ids import require_uuid

        with pytest.raises(GoodMemError, match=r"^memory_id must be a UUID"):
            require_uuid(bad, "memory_id")
        assert require_uuid(VICTIM.upper(), "memory_id") == VICTIM
        assert require_uuid(uuid.UUID(VICTIM), "memory_id") == VICTIM


# ---------------------------------------------------------------------------
# Ids that pass the check as one value and are sent as another
# ---------------------------------------------------------------------------
#
# The first version of the check validated the caller's object but returned
# str(value) for a uuid.UUID and value.lower() for a str -- both of which a
# subclass can override. client.delete_memory(_UUIDWithLyingStr(VICTIM)) passed
# the check and still sent DELETE /v1/spaces/<DECOY>. JSON, model output and
# environment variables only ever produce a plain str, so this needs code in
# the same process; the id that is sent must still be the one that was checked.

TRAVERSAL = f"../spaces/{VICTIM}"
# What the lying methods answer. A different space from VICTIM, so that for a
# space_id argument the traversal cannot land on the intended path by luck.
DECOY = "0199a8c0-5e1f-7c3a-9d2b-4f6e8a1b2c3d"
LIE = f"../spaces/{DECOY}"


class _UUIDWithLyingStr(uuid.UUID):
    def __str__(self) -> str:
        return LIE

    def __format__(self, spec: str) -> str:
        return LIE

    @property
    def hex(self) -> str:  # type: ignore[override]
        return LIE


class _UUIDWithLyingInt(uuid.UUID):
    @property
    def int(self) -> int:  # type: ignore[override]
        return -1


def _uuid_with_lying_int() -> uuid.UUID:
    # uuid.UUID.__init__ cannot assign through the property, so fill the
    # slot the way __init__ would.
    made = object.__new__(_UUIDWithLyingInt)
    vars(uuid.UUID)["int"].__set__(made, uuid.UUID(VICTIM).int)
    object.__setattr__(made, "is_safe", uuid.SafeUUID.unknown)
    return made


class _StrWithLyingMethods(str):
    def lower(self) -> str:  # type: ignore[override]
        return LIE

    def __str__(self) -> str:
        return LIE

    def __format__(self, spec: str) -> str:
        return LIE


class _StrThatClaimsToBeTheVictim(str):
    """Holds a traversal, but every overridable method answers VICTIM."""

    def lower(self) -> str:  # type: ignore[override]
        return VICTIM

    def __str__(self) -> str:
        return VICTIM

    def __format__(self, spec: str) -> str:
        return VICTIM

    def __repr__(self) -> str:
        return repr(VICTIM)


class _PosesAsUUID:
    """Not a UUID at all, but isinstance(x, uuid.UUID) answers True."""

    @property  # type: ignore[misc]
    def __class__(self):  # type: ignore[override]
        return uuid.UUID

    def __str__(self) -> str:
        return TRAVERSAL


class _PosesAsStr:
    """Not a str at all, but isinstance(x, str) answers True."""

    @property  # type: ignore[misc]
    def __class__(self):  # type: ignore[override]
        return str

    def __str__(self) -> str:
        return TRAVERSAL


class _ReprRaises:
    def __repr__(self) -> str:
        raise RuntimeError("repr refused")


# The real id sits in each of these; only its methods lie about it.
HONEST_DATA = {
    "uuid-str-lies": lambda: _UUIDWithLyingStr(VICTIM),
    "uuid-int-lies": _uuid_with_lying_int,
    "str-methods-lie": lambda: _StrWithLyingMethods(VICTIM),
    "str-methods-lie-upper": lambda: _StrWithLyingMethods(VICTIM.upper()),
}

# None of these holds a UUID, whatever its methods or its __class__ say.
DISHONEST_DATA = {
    "poses-as-uuid": _PosesAsUUID,
    "poses-as-str": _PosesAsStr,
    "str-holding-traversal": lambda: _StrThatClaimsToBeTheVictim(TRAVERSAL),
    "uninitialised-uuid": lambda: object.__new__(uuid.UUID),
    "repr-raises": _ReprRaises,
}


class TestTheIdSentIsTheIdChecked:
    @pytest.mark.parametrize("make", list(HONEST_DATA))
    @pytest.mark.parametrize("label", list(ENTRY_POINTS))
    def test_a_subclass_sends_its_own_data_not_what_its_methods_say(
        self, recorded, label, make
    ):
        client, requests = recorded
        _, _, call, expected = ENTRY_POINTS[label]
        result, exc = _call(lambda: call(client, HONEST_DATA[make]()))
        sent = [(verb, urlsplit(path).path) for verb, path, _ in requests]
        assert sent == [
            (verb, path.format(u=VICTIM)) for verb, path in expected
        ], f"{label} with {make} sent {sent}; it returned {result!r} / {exc!r}"
        assert exc is None
        for _, path, body in requests:
            assert DECOY not in path and DECOY.encode() not in body
            assert b".." not in body and b"/spaces/" not in body
        if expected[-1][0] == "POST":
            # A body-carried id is the real one, not what a method made up.
            assert VICTIM.encode() in requests[-1][2]

    @pytest.mark.parametrize("make", list(DISHONEST_DATA))
    @pytest.mark.parametrize("label", list(ENTRY_POINTS))
    def test_an_object_that_lies_about_its_type_is_refused(self, recorded, label, make):
        client, requests = recorded
        _, param, call, _ = ENTRY_POINTS[label]
        result, exc = _call(lambda: call(client, DISHONEST_DATA[make]()))
        sent = [f"{verb} {path}" for verb, path, _ in requests]
        assert sent == [], f"{label} with {make} reached the server as {sent}"
        assert isinstance(exc, GoodMemError), f"not refused cleanly: {exc!r}"
        assert param in str(exc) and "must be a UUID" in str(exc)
        assert result is None

    @pytest.mark.parametrize("make", list(HONEST_DATA))
    def test_the_validator_returns_a_new_exact_str(self, make):
        from honeyhive_goodmem._ids import require_uuid, require_uuids

        given = HONEST_DATA[make]()
        out = require_uuid(given, "memory_id")
        assert type(out) is str and out == VICTIM
        (listed,) = require_uuids([given], "space_ids")
        assert type(listed) is str and listed == VICTIM
        (single,) = require_uuids(given, "space_ids")
        assert type(single) is str and single == VICTIM

    @pytest.mark.parametrize("make", list(DISHONEST_DATA))
    def test_the_validator_refuses_with_its_own_error(self, make):
        from honeyhive_goodmem._ids import require_uuid

        with pytest.raises(GoodMemError, match=r"^memory_id must be a UUID"):
            require_uuid(DISHONEST_DATA[make](), "memory_id")

    def test_a_uuid_whose_stored_value_is_out_of_range_is_refused(self):
        from honeyhive_goodmem._ids import require_uuid

        broken = uuid.UUID(VICTIM)
        object.__setattr__(broken, "int", 1 << 130)
        with pytest.raises(GoodMemError, match=r"^memory_id must be a UUID"):
            require_uuid(broken, "memory_id")


# ---------------------------------------------------------------------------
# With a HoneyHive tracer active
# ---------------------------------------------------------------------------
#
# Every method is wrapped in honeyhive's @trace. With a tracer active that
# wrapper records the call's inputs, and it runs the method a second time,
# untraced, when the exception text contains "Tracer error". A refusal must
# still send nothing, reach the caller, and show up in the trace as an error.


@pytest.fixture(scope="class")
def honeyhive_spans(recording_server):
    """An offline HoneyHive tracer whose spans land in memory."""
    from honeyhive import HoneyHiveTracer
    from honeyhive.tracer import registry
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    host, port = recording_server.server_address
    tracer = HoneyHiveTracer.init(
        api_key="hh_offline_test_key",
        # Anything HoneyHive itself sent would be recorded -- and counted.
        server_url=f"http://{host}:{port}",
        test_mode=True,
        disable_batch=True,
    )
    exporter = InMemorySpanExporter()
    tracer.provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    registry.clear_registry()
    exporter.shutdown()


# retrieve() returns a structured outcome and is not traced itself.
TRACED = [label for label, (method, *_) in ENTRY_POINTS.items() if method != "retrieve"]


@pytest.mark.filterwarnings(
    "ignore:You should use instrumentation_scope:DeprecationWarning"
)
class TestRefusalsUnderAnActiveTracer:
    @pytest.mark.parametrize(
        "bad", [TRAVERSAL, f"Tracer error/../../spaces/{VICTIM}"], ids=repr
    )
    @pytest.mark.parametrize("label", TRACED)
    def test_a_refusal_sends_nothing_and_is_an_error_span(
        self, honeyhive_spans, recorded, label, bad
    ):
        client, requests = recorded
        honeyhive_spans.clear()
        method, param, call, _ = ENTRY_POINTS[label]
        result, exc = _call(lambda: call(client, bad))
        assert [f"{verb} {path}" for verb, path, _ in requests] == []
        assert isinstance(exc, GoodMemError) and param in str(exc)
        assert result is None

        spans = {s.name: s for s in honeyhive_spans.get_finished_spans()}
        event = f"goodmem.{method}"
        assert event in spans, f"no span for {event}: {sorted(spans)}"
        assert spans[event].status.status_code.name == "ERROR"
        recorded_types = {
            str((e.attributes or {}).get("exception.type", ""))
            for e in spans[event].events
            if e.name == "exception"
        }
        assert any(t.endswith("GoodMemError") for t in recorded_types)
        if "Tracer error" in bad:
            # honeyhive ran the method again, untraced, and it refused again:
            # still nothing sent, and no separate error span.
            assert f"{event}_error" not in spans
        else:
            error = spans[f"{event}_error"].attributes or {}
            assert error.get("honeyhive_error_type") == "GoodMemError"
            assert "must be a UUID" in str(error.get("honeyhive_error"))

    @pytest.mark.parametrize("label", TRACED)
    def test_a_valid_id_is_an_ordinary_span(self, honeyhive_spans, recorded, label):
        client, requests = recorded
        honeyhive_spans.clear()
        method, _, call, expected = ENTRY_POINTS[label]
        call(client, VICTIM)
        sent = [(verb, urlsplit(path).path) for verb, path, _ in requests]
        assert sent == [(verb, path.format(u=VICTIM)) for verb, path in expected]
        spans = {s.name: s for s in honeyhive_spans.get_finished_spans()}
        assert spans[f"goodmem.{method}"].status.status_code.name != "ERROR"
        assert not [name for name in spans if name.endswith("_error")]


# ---------------------------------------------------------------------------
# Packaging and README facts
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


class TestPackaging:
    def test_requests_is_declared(self):
        """honeyhive 1.6.0 imports requests at import time without declaring it.

        It used to arrive through opentelemetry-exporter-otlp-proto-http,
        which made it an optional extra in 1.45.0; after that, CI's own
        install (pip install -e . pytest httpx ...) could not import this
        package at all.
        """
        import tomllib

        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        names = {
            re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0].lower()
            for requirement in project["dependencies"]
        }
        assert "requests" in names

    def test_readme_names_the_license_that_ships(self):
        license_name = (ROOT / "LICENSE").read_text().splitlines()[0]
        assert license_name == "MIT License"
        pyproject = (ROOT / "pyproject.toml").read_text()
        assert "License :: OSI Approved :: MIT License" in pyproject
        section = (ROOT / "README.md").read_text().split("\n## License\n", 1)[1]
        assert "MIT" in section and "Apache" not in section


class TestOptionalReranker:
    def test_none_means_no_reranker(self, recorded):
        client, requests = recorded
        client.retrieve_memories("q", [SPACE_ID], reranker_id=None)
        (body,) = [json.loads(b) for _, _, b in requests]
        assert "reranker" not in json.dumps(body).lower()

    def test_an_empty_string_is_refused_not_read_as_none(self, recorded):
        """0.2.0 treated "" as "no reranker"; it is now a non-UUID id."""
        client, requests = recorded
        with pytest.raises(GoodMemError, match=r"^reranker_id must be a UUID"):
            client.retrieve_memories("q", [SPACE_ID], reranker_id="")
        assert requests == []
