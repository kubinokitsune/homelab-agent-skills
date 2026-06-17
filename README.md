# Homelab Agent — Shared Skills Library

The shared Python library every homelab agent (Forge, Mason, Apex, Eos, Hermes,
Warden, Iris, Axiom, Kairos, Scout) builds on. Cross-cutting concerns are
written once here so they behave identically everywhere.

## The Step 0 contract

Before any actual skill, four foundations were locked. Every skill is written
against these:

| File | Role |
|------|------|
| `skills/config.py`  | One `config` object. Skills read hosts/secrets from it, never `os.getenv`. Validation is lazy and per-skill: a missing key only fails when a skill actually uses it, with a message naming the key and the file to fix. |
| `skills/result.py`  | The return shape. I/O skills return `Result.success(data)` or `Result.failure("reason")` — never bare booleans, never raising into the agent. |
| `skills/logging.py` | `get_logger(__name__)`. Every line is `date time  LEVEL  [Agent]  skill  message`. Stdout only (container/Loki friendly). Agent tag set via `set_agent("Forge")` at startup. |
| `skills/errors.py`  | The `@skill` decorator: any exception a skill raises is logged with a traceback and converted to `Result.failure(...)`, so a dead service never crashes an agent. |

## Skill template

```python
from skills.config import config
from skills.logging import get_logger
from skills.errors import skill
from skills.result import Result

log = get_logger(__name__)

@skill
def do_something(...) -> Result:
    host = config.qdrant_host          # config, never os.getenv
    ...                                # happy path only
    return Result.success(payload)     # failures auto-handled by @skill
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in real values (never committed)
```

## Global skills

| Skill | Does | External dep |
|-------|------|--------------|
| `notifications.py` | Discord + Pushover alerts, routed by severity | webhooks, Pushover |
| `obsidian_vault.py` | Read/write/search the vault (frontmatter-aware, no-clobber) | the vault folder |
| `vector_store.py`   | Semantic memory: upsert/query/delete/count | Qdrant + Ollama |
| `web_search.py`     | Keyless web search, normalized results | DuckDuckGo (ddgs) |
| `file_ops.py`       | Allowlisted server filesystem ops | `FILE_OPS_ROOTS` |

## Status

Phase 1 skills complete (all tested against live infra). Deferred to Phase 2
(need server infra): `health_data`, `calendar`, `git_ops`, `homelab_api`, `mqtt_client`.
