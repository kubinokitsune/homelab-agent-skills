"""print_tuning -- Mason's ledger of print-quality recommendations.

Each ``!assess`` (or dropped photo) produces a diagnosis + fixes. They're tracked
here as pending -> applied, so improvements actually carry into the NEXT print
instead of being forgotten. Live-settable fixes (temp/flow/fan/speed/PA) carry a
structured ``actions`` list Mason can apply on the printer; slicer fixes are
reminders Pipe marks done by hand.

Plain JSON, no deps. Call from the async agent via asyncio.to_thread.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger("print_tuning")


def _load() -> list[dict]:
    p = config.tuning_path
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(items: list[dict]) -> None:
    p = config.tuning_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, indent=2), encoding="utf-8")


@skill
def add(defect: str, advice: str, actions: list | None = None,
        kind: str = "mixed", filament: str = "") -> Result:
    """Log a recommendation. ``actions`` = [{key,value}] live-settable changes."""
    items = _load()
    entry = {
        "id": uuid.uuid4().hex[:4],
        "ts": datetime.now().isoformat(timespec="seconds"),
        "defect": defect.strip()[:80] or "print quality",
        "advice": advice.strip(),
        "actions": actions or [],
        "kind": kind,                 # live | slicer | mixed
        "filament": filament,
        "status": "pending",          # pending | applied | dismissed
    }
    items.append(entry)
    _save(items)
    return Result.success(entry)


@skill
def pending() -> Result:
    return Result.success([e for e in _load() if e.get("status") == "pending"])


@skill
def get(entry_id: str) -> Result:
    for e in _load():
        if e["id"] == entry_id:
            return Result.success(e)
    return Result.failure(f"no tuning entry `{entry_id}`")


@skill
def set_status(entry_id: str, status: str) -> Result:
    items = _load()
    for e in items:
        if e["id"] == entry_id:
            e["status"] = status
            _save(items)
            return Result.success(e)
    return Result.failure(f"no tuning entry `{entry_id}`")


@skill
def history(n: int = 10) -> Result:
    return Result.success(list(reversed(_load()))[:n])
