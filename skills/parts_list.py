"""Running engineering parts / shopping list -- Forge's master-doc feature #5.

JSON-backed (reliable CRUD) under ``config.parts_list_path``; Forge renders a
human-readable checklist note into the vault on each change, so it shows in
Obsidian and Axiom indexes it.

    parts_list.add("M3x10 bolts", qty=10, notes="F1 car")
    parts_list.items()              # open items
    parts_list.complete(item_id)    # mark bought
"""

from __future__ import annotations

import json
import uuid
from datetime import date
from pathlib import Path

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)


def _path() -> Path:
    return config.parts_list_path


def _load() -> list[dict]:
    p = _path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save(items: list[dict]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


@skill
def add(item: str, qty: int = 1, notes: str = "") -> Result:
    """Add an item to the shopping list. Returns the entry (with its short id)."""
    item = item.strip()
    if not item:
        return Result.failure("need an item to add")
    try:
        qty = max(1, int(qty))
    except (TypeError, ValueError):
        qty = 1
    items = _load()
    entry = {
        "id": uuid.uuid4().hex[:4],
        "item": item, "qty": qty, "notes": notes.strip(),
        "status": "open", "added": date.today().isoformat(),
    }
    items.append(entry)
    _save(items)
    log.info("parts add %dx %s", qty, item)
    return Result.success(entry)


@skill
def items(include_done: bool = False) -> Result:
    """Open items (newest last). Set ``include_done`` for the full list."""
    data = _load()
    if not include_done:
        data = [e for e in data if e.get("status") != "bought"]
    return Result.success(data)


@skill
def complete(item_id: str) -> Result:
    """Mark an item bought. Returns the updated entry."""
    data = _load()
    found = None
    for e in data:
        if e["id"] == item_id:
            e["status"] = "bought"
            e["bought"] = date.today().isoformat()
            found = e
    if not found:
        return Result.failure(f"no item with id {item_id}")
    _save(data)
    return Result.success(found)


@skill
def remove(item_id: str) -> Result:
    """Delete an item entirely (not the same as marking it bought)."""
    data = _load()
    kept = [e for e in data if e["id"] != item_id]
    if len(kept) == len(data):
        return Result.failure(f"no item with id {item_id}")
    _save(kept)
    return Result.success({"removed": item_id})
