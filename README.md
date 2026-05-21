# honeyhive-goodmem

[GoodMem](https://goodmem.ai) integration for
[HoneyHive](https://honeyhive.ai).

This package wraps the GoodMem REST API with a Python client whose every
operation is decorated with HoneyHive's `@trace`, so calls show up as spans in
the active HoneyHive session. The operation shapes mirror the reference
GoodMem integration so behavior is identical across frameworks.

## Installation

```bash
pip install honeyhive-goodmem
```

For local development:

```bash
pip install -e .
```

## Quickstart

```python
from honeyhive import HoneyHiveTracer
from honeyhive_goodmem import GoodMemClient, GoodMemConfig

HoneyHiveTracer.init(api_key="hh_...", project="my-project")

client = GoodMemClient(
    GoodMemConfig(
        base_url="https://localhost:8080",
        api_key="gm_xxxxxxxxxxxxxxxxxxxxxxxx",
        verify_ssl=False,  # self-signed local server
    )
)

embedders = client.list_embedders()
embedder_id = embedders["embedders"][0]["embedder_id"]

space = client.create_space(name="quickstart", embedder_id=embedder_id)
space_id = space["space_id"]

client.create_memory(
    space_id=space_id,
    text_content="The capital of France is Paris.",
)

results = client.retrieve_memories(
    query="What is the capital of France?",
    space_ids=[space_id],
    max_results=3,
)
print(results)
```

## Available operations

`GoodMemClient` exposes the following 11 traced methods, matching the
reference GoodMem integration:

| Method | Description |
|--------|-------------|
| `list_embedders` | List embedder models available on the server |
| `list_spaces` | List all spaces accessible to the API key |
| `get_space` | Fetch a space by ID |
| `create_space` | Create a space (idempotent by name) |
| `update_space` | Update a space's name / labels / public-read flag |
| `delete_space` | Delete a space |
| `create_memory` | Store text or a file as a memory |
| `list_memories` | List memories in a space |
| `retrieve_memories` | Semantic retrieval, with optional reranker / LLM |
| `get_memory` | Fetch a memory by ID (with original content) |
| `delete_memory` | Delete a memory |

### Retrieval options

`GoodMemClient.retrieve_memories` accepts the GoodMem post-processor
parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `reranker_id` | UUID | Reranker model to improve result ordering |
| `llm_id` | UUID | LLM used to generate a contextual abstract reply |
| `relevance_threshold` | 0–1 | Minimum score for including a result |
| `llm_temperature` | 0–2 | Creativity for the LLM post-processor |
| `max_results` | int | Cap on returned chunks |
| `chronological_resort` | bool | Reorder results by memory creation time |

## License

MIT
