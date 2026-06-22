"""Error / confusion log -- the record layer for pattern clustering.

An agent jots down a problem it (or Pipe) hit; later the embeddings of these
entries are clustered (see ``vector_store.cluster``) to surface *recurring*
problems -- the master doc's "clustering similar errors → pattern recognition".

File-based JSONL, one file per agent under ``config.error_log_dir`` (machine
output, outside the vault), so it syncs to the server later like agent-reports.

    error_log.log("Forge", "I2C scan finds nothing on the SSD1306", source="manual")
    entries = error_log.load("Forge")            # open entries, newest first
    error_log.resolve("Forge", entry_id)         # mark one solved
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

_log = get_logger(__name__)  # underscored: the public skill below is named log()


def _path(agent: str) -> Path:
    return config.error_log_dir / f"{agent}.jsonl"


@skill
def log(agent: str, text: str, source: str = "manual", context: str = "") -> Result:
    """Append an error/confusion entry. Returns the entry (with its id)."""
    text = (text or "").strip()
    if not text:
        return Result.failure("empty error text")
    entry = {
        "id": datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        "time": datetime.now().isoformat(timespec="seconds"),
        "text": text,
        "source": source,      # manual | review | auto
        "context": context,
        "status": "open",
    }
    path = _path(agent)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _log.info("logged %s error %s: %s", agent, entry["id"], text[:60])
    return Result.success(entry)


def _read_all(agent: str) -> list[dict]:
    path = _path(agent)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    return entries


@skill
def load(agent: str, include_resolved: bool = False) -> Result:
    """Return entries (newest first). Open-only unless ``include_resolved``."""
    entries = _read_all(agent)
    if not include_resolved:
        entries = [e for e in entries if e.get("status") != "resolved"]
    entries.sort(key=lambda e: e.get("id", ""), reverse=True)
    return Result.success(entries)


@skill
def resolve(agent: str, entry_id: str) -> Result:
    """Mark an entry resolved (rewrites the file). Returns the updated entry."""
    entries = _read_all(agent)
    found = None
    for e in entries:
        if e.get("id") == entry_id:
            e["status"] = "resolved"
            found = e
    if not found:
        return Result.failure(f"no error with id {entry_id}")
    path = _path(agent)
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    _log.info("resolved %s error %s", agent, entry_id)
    return Result.success(found)
