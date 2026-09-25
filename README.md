# honeyhive-goodmem

[GoodMem](https://docs.goodmem.ai) memory as [HoneyHive](https://honeyhive.ai)-traced
operations. Every call appears as a span alongside the rest of your agent's
work, so memory reads and writes are visible in the same trace as the model
calls they feed.

**Version 0.2.1.** Verified against GoodMem server **v1.0.320**.

> **Upgrading from 0.1.0.** This is an observability package, which makes
> 0.1.0's worst defect specific to it: retrieval statuses were dropped, so a
> retrieval that **failed** was recorded as a **successful span**. A space
> whose embedder was unavailable produced `success: true, totalResults: 0` —
> in HoneyHive that reads as "the index is empty", not "the search broke".
> See [Changes in 0.2.0](#changes-in-020).

> **Security.** 0.1.0's test file carried a live GoodMem API key as a default
> value, on the public default branch and in the `v0.1.0` tag. It is removed
> here and the environment variable is now required with no fallback — but
> removing it from the tree does not un-leak it. **That key needs rotating.**

> **Security (0.2.1).** Every id argument must now be a UUID. In 0.2.0 the
> GoodMem SDK put ids into URL paths raw, so `delete_memory("../spaces/<id>")`
> sent `DELETE /v1/spaces/<id>`, deleted a whole space and returned
> `success: True`. And `GoodMemConfig` printed its API key: HoneyHive's
> `@trace` records a traced function's arguments with `str()`, so a function
> that took the config exported the key to HoneyHive. It is now masked
> everywhere. See [Changes in 0.2.1](#changes-in-021).

## Install

```bash
pip install honeyhive-goodmem
```

## Use

```python
from honeyhive import HoneyHiveTracer
from honeyhive_goodmem import GoodMemClient, GoodMemConfig

HoneyHiveTracer.init(api_key="<your-honeyhive-key>", project="my-project")

client = GoodMemClient(
    GoodMemConfig(base_url="https://your-goodmem-server", api_key="<your-goodmem-key>")
)
```

Both GoodMem settings fall back to `GOODMEM_BASE_URL` and `GOODMEM_API_KEY`.

`GoodMemConfig` stores the key as a `SecretStr` (a `str` or a
`pydantic.SecretStr` is accepted too). `repr()`, `str()`, f-strings, logging,
`dataclasses.asdict()` and `json.dumps(..., default=str)` all show
`**********`, and so does the span of a `@trace`-decorated function of yours
that takes the config as an argument. `GoodMemClient` does not keep the key
itself; it goes straight to the SDK, which sends it as `X-API-Key`. When you
need the raw value, ask for it:

```python
config = GoodMemConfig(base_url="https://your-goodmem-server", api_key="<your-goodmem-key>")
print(config)                  # GoodMemConfig(..., api_key=SecretStr('**********'), ...)
raw = config.get_api_key()     # or config.api_key.get_secret_value()
```

`config.api_key` is no longer a `str`, so passing it straight to an HTTP
library raises `TypeError` rather than sending the mask; use `get_api_key()`.
For the same reason `config.api_key == "<your-goodmem-key>"` is always
`False`; compare `config.get_api_key()` instead.

Without a tracer the methods still run; HoneyHive logs that no tracer is
active and no span is emitted. With a tracer, a call that raises
`GoodMemError` — a refused id included — is an error span carrying the
exception.

## What a retrieval span records

```python
client.retrieve_memories("what did I store?", space_ids=["<space-uuid>"], max_results=5)
```

```python
{
  "success": True,
  "query": "...",
  "results": [
    {
      "chunk_id": "...", "chunk_text": "...", "memory_id": "...", "space_id": "...",
      "score": 0.64,          # higher is better
      "raw_score": -0.64,     # exactly what the server sent
      "score_kind": "vector", # or "reranker" -- not the same scale
      "content_type": "text/plain",
      "metadata": {...},      # the memory's metadata, joined by UUID
    }
  ],
  "total_results": 1,
  "partial": False,           # True when the server reported a problem
  "statuses": [],             # what it reported
  "result_set_id": "...",
}
```

`partial` means exactly one thing: **the server reported a real problem during
this retrieval.** It is independent of whether hits came back. A degraded
search still returns whatever arrived, with `partial` set and a `warning` key;
when nothing usable arrives the result is empty and still flagged. The span
therefore shows a failed retrieval as failed.

For structured use, `client.retrieve(...)` returns the same data as a
`RetrievalOutcome` object instead of a traced dictionary.

### Scores

GoodMem produces two kinds of score and they are not comparable. **Vector**
scores are negative distances, so `score` is the flipped value with
`raw_score` kept beside it. **Reranker** scores are already higher-is-better,
on a **provider-dependent** scale — measured live on the same five documents,
Voyage `rerank-2.5` returned `0.27..0.93` and Jina `jina-reranker-v3` returned
`-0.14..0.43`. There is therefore no default threshold anywhere in this
package.

`score_kind` records what the server did, not what was asked for. When a
requested reranker fails, the server reports `RERANKING_FAILED` (and
`NOT_FOUND` for a reranker id it cannot find) and still returns the
vector-stage hits. Those hits are labelled `"vector"` and flipped like any
other vector score, and the retrieval is `partial` with both statuses.

## Metadata filters

Filters are expressions evaluated server-side, not SQL:

```python
from honeyhive_goodmem import filters

client.retrieve_memories("q", ["<space-uuid>"], metadata_filter={"tenant": "acme"})

expression = filters.all_of(
    filters.equals("tenant", "acme"),
    filters.compare("year", ">=", 2026),
)
```

The helper applies the escaping the server accepts (`'` → `\'`, `\` → `\\`;
SQL-style `''` doubling is rejected with HTTP 400), refuses control
characters, restricts field names, and casts each value to the type GoodMem
stored — a boolean compared as `TEXT` is accepted with HTTP 200 and matches
nothing.

## Operations

| Method | Event |
| --- | --- |
| `create_space`, `list_spaces`, `get_space`, `update_space`, `delete_space` | `tool` |
| `list_embedders` | `tool` |
| `create_memory`, `get_memory`, `list_memories`, `delete_memory` | `tool` |
| `retrieve_memories` | `retrieval` |

`update_space` takes `name` and `labels`. It no longer offers `public_read`:
the server removed that field and answers `400 Unrecognized field "publicRead"`.

Every id argument (`space_id`, `memory_id`, `embedder_id`, `space_ids`,
`reranker_id`) must be a UUID, because the SDK places ids into URL paths
unescaped and a value such as `../spaces/<id>` would otherwise address a
different resource. Anything else raises `GoodMemError` before a request is
made; upper-case UUIDs and `uuid.UUID` objects are accepted and sent in
lower case. What is sent is a new plain string built from the id's own
characters (or a `uuid.UUID`'s stored value), never the object passed in, so
a `str` or `uuid.UUID` subclass cannot change it through `lower()` or
`__str__`. `reranker_id` is optional: leave it `None` for no reranker. An
empty string is refused like any other non-UUID, so
`os.getenv("RERANKER_ID", "")` needs `or None`.

## Changes in 0.2.1

Measured against a local server that records every request, driving the real
SDK and `httpx`:

| Was (0.2.0) | Now |
| --- | --- |
| `delete_memory("../spaces/<id>")` sent `DELETE /v1/spaces/<id>` and returned `success: True` — a whole space deleted through a memory call | Refused with `GoodMemError: memory_id must be a UUID ...`; nothing is sent |
| `a/../../spaces/<id>` and `<id>/../../spaces/<id>` were resolved the same way by `get_space`, `update_space`, `delete_space`, `list_memories`, `get_memory`, `delete_memory` | Refused, every entry point |
| Other malformed ids were sent too: `%2e%2e/spaces/<id>` and `..%2Fspaces%2F<id>` verbatim (the GoodMem server normalises `%2e%2e` into a traversal), `" <id>"` as `%20<id>`, `<id>?x=1` with a query string, `<id>#frag` as `<id>`, `None` as `/None` | Refused |
| Non-UUID `space_ids`, `embedder_id` and `reranker_id` reached request bodies | Refused, by the same check |
| 145 of the 166 bad-id cases in the new regression suite reached the server; 136 of them came back as success | 0 reach the server |
| `retrieve_memories(..., reranker_id="")` meant "no reranker" | Refused as a non-UUID id; pass `None` (the default) instead |
| A `uuid.UUID` or `str` subclass whose `__str__`, `__format__` or `lower()` returned `../spaces/<id>`, or an object whose `__class__` property claimed to be `uuid.UUID`, was sent as that path: `delete_memory` sent `DELETE /v1/spaces/<id>` and returned `success: True`. The first draft of this fix checked such an object but still sent what its methods returned | The id sent is a new plain string rebuilt from the characters or stored value that were checked, then checked again; such an object is sent as its real id or refused |
| `pip install -e .` into a fresh environment could not `import honeyhive_goodmem`: `honeyhive` imports `requests` without declaring it, and `opentelemetry-exporter-otlp-proto-http` 1.45.0 stopped pulling it in | `requests` is declared here |
| `GoodMemConfig` held `api_key` as a plain `str`, so `repr()`, `str()`, f-strings, `logger.warning("%s", config)` and `dataclasses.asdict()` all printed the key. A `@trace`-decorated function taking the config exported it to HoneyHive as the span attribute `honeyhive_inputs.config`. `GoodMemClient` also kept the raw key in `_GoodMemClient__api_key`, so `json.dumps(vars(client), default=str)` contained it | The key is held in a `SecretStr` that renders as `**********` in every one of those, including the span. The client keeps no copy. The server still receives the real key in `X-API-Key`; read it yourself with `config.get_api_key()` |
| With a `reranker_id` whose reranker failed, the server's vector fallback hits were labelled `score_kind: "reranker"` and left un-negated, so a `-0.58` distance was reported as `score: -0.58` | `score_kind` comes from the response: after `RERANKING_FAILED` or a reranker `NOT_FOUND` the hits are `"vector"` and scored `0.58`. `partial` and both statuses are unchanged |
| With a HoneyHive tracer active, a write that failed with a server message containing `Tracer error` was sent twice: honeyhive's `@trace` re-runs the function, untraced, whenever the error text contains that phrase, and the server can echo request content. Measured: `create_memory` and `create_space` sent 2 POSTs, `delete_memory` and `delete_space` 2 DELETEs | Sent once. Every method runs at most once per call; a re-run returns the first result or re-raises the first error. Tracing is unchanged |
| A `uuid.UUID` whose stored `int` raised from `__index__`, or an iterable of space ids whose `__iter__` raised, escaped as `RuntimeError`/`KeyError` instead of `GoodMemError` (nothing was sent) | `GoodMemError`, before any request |
| `config.api_key` was typed `str \| SecretStr` although it is always a `SecretStr`, so a typed caller of `config.api_key.get_secret_value()` got a mypy error | Typed `SecretStr`; the constructor still accepts a `str` |

## Changes in 0.2.0

Reproduced against the published 0.1.0 wheel, live against GoodMem v1.0.320.

| Was | Now |
| --- | --- |
| A live GoodMem API key was the default value of `GOODMEM_API_KEY` in the test file, on the public default branch | Removed; the variable is required with no fallback. **The key still needs rotating** |
| A failing embedder produced `success: true, totalResults: 0` — the server's `EMBEDDER_FAILED` was dropped, so the span said success | `partial` + `statuses` + `warning` in the traced payload |
| A broken reranker produced `success: true` with three statuses discarded | Same contract; hits are still returned, flagged |
| Hand-written `httpx` client | Official `goodmem` SDK |
| `public_read` was a parameter; the server answers HTTP 400 | Gone |
| Empty search took **11.65 s** — `wait_for_indexing` on by default | **0.31 s**; the read path never polls |
| Reusing a space name reported the embedder you asked for while the space ran another | Reuse requires a match; a mismatch names both |
| Chunks and memories were two arrays joined by positional `memory_index` | Joined by UUID, de-duplicated by chunk id |
| Raw negative scores in the span | `score` / `raw_score` / `score_kind` |
| `nextToken` appeared nowhere — listings returned one page | Paginated, bounded by `max_list_items` |
| No metadata filtering | `filters`, escaped and type-correct |
| The API key was a public attribute | Private; absent from `repr` and from every traced payload |
| 13 live-only tests that fell back to a committed key; no CI | 33 offline + 16 live; CI on 3.11–3.13 |

Already correct in 0.1.0 and unchanged: request timeouts (30 s by default),
and the error path — the server's own message reaches the caller.

## Tests

| Suite | Count | Needs |
| --- | --- | --- |
| `tests/test_honeyhive_goodmem.py` | 451 | nothing — the real SDK over a mock transport or a local server that records every request, fed JSON and NDJSON captured from a live server; the span tests use HoneyHive's own tracer in test mode with an in-memory exporter |
| `tests/test_honeyhive_goodmem_live.py` | 16 | `GOODMEM_API_KEY` + `GOODMEM_BASE_URL`; skips entirely without them |

```bash
pip install -e . pytest httpx "ruff==0.7.4" mypy

pytest tests/test_honeyhive_goodmem.py

GOODMEM_API_KEY=... GOODMEM_BASE_URL=... \
  GOODMEM_TEST_EMBEDDER_ID=... \
  pytest tests/test_honeyhive_goodmem_live.py

ruff check honeyhive_goodmem tests && ruff format --check honeyhive_goodmem tests
mypy honeyhive_goodmem
```

One offline test scans the tree for a credential-shaped string, so the defect
that shipped in 0.1.0 cannot come back unnoticed. The live suite creates one
space per run and asserts, against a fresh listing, that it is gone.

## License

MIT — see [LICENSE](LICENSE).
