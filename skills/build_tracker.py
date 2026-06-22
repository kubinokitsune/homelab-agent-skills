"""Build step-tracking -- Forge's master-doc feature #4.

Track a build as an ordered checklist; mark steps done hands-free as you go. One
active build at a time; JSON-backed (config.build_tracker_path) so it survives
restarts. Forge mirrors the active build into a vault checklist note.

    build_tracker.start("F1 wing mount", ["print mount", "tap M3 holes", "fit"])
    build_tracker.mark_done()      # completes the first not-done step
    build_tracker.active()
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)


def _path() -> Path:
    return config.build_tracker_path


def _load() -> dict:
    p = _path()
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, dict) and "builds" in d:
                return d
        except (OSError, ValueError):
            pass
    return {"active": None, "builds": {}}


def _save(d: dict) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def _with_name(build: dict, name: str) -> dict:
    return {"name": name, **build}


@skill
def start(name: str, steps: list[str] | None = None) -> Result:
    """Create (or replace) a build and make it active."""
    name = name.strip()
    if not name:
        return Result.failure("a build needs a name")
    d = _load()
    d["builds"][name] = {
        "created": date.today().isoformat(),
        "steps": [{"text": s.strip(), "done": False} for s in (steps or []) if s.strip()],
    }
    d["active"] = name
    _save(d)
    log.info("build start '%s' (%d steps)", name, len(d["builds"][name]["steps"]))
    return Result.success(_with_name(d["builds"][name], name))


@skill
def set_active(name: str) -> Result:
    name = name.strip()
    d = _load()
    if name not in d["builds"]:
        return Result.failure(f"no build named '{name}'")
    d["active"] = name
    _save(d)
    return Result.success(_with_name(d["builds"][name], name))


@skill
def active() -> Result:
    """The active build (with its name), or None if there isn't one."""
    d = _load()
    name = d.get("active")
    if not name or name not in d["builds"]:
        return Result.success(None)
    return Result.success(_with_name(d["builds"][name], name))


@skill
def add_step(text: str) -> Result:
    text = text.strip()
    if not text:
        return Result.failure("step needs text")
    d = _load()
    name = d.get("active")
    if not name:
        return Result.failure("no active build -- start one first")
    d["builds"][name]["steps"].append({"text": text, "done": False})
    _save(d)
    return Result.success(_with_name(d["builds"][name], name))


@skill
def mark_done(index: int | None = None) -> Result:
    """Mark the first not-done step done (or a specific 0-based index)."""
    d = _load()
    name = d.get("active")
    if not name:
        return Result.failure("no active build")
    steps = d["builds"][name]["steps"]
    if index is None:
        index = next((i for i, s in enumerate(steps) if not s["done"]), None)
        if index is None:
            return Result.failure("every step is already done")
    if not 0 <= index < len(steps):
        return Result.failure("step number out of range")
    steps[index]["done"] = True
    _save(d)
    return Result.success({"name": name, "completed": steps[index]["text"], "steps": steps})


@skill
def builds() -> Result:
    """Summaries of all builds: name, active flag, done/total."""
    d = _load()
    out = [
        {"name": n, "active": n == d.get("active"),
         "done": sum(1 for s in b["steps"] if s["done"]), "total": len(b["steps"])}
        for n, b in d["builds"].items()
    ]
    return Result.success(out)
