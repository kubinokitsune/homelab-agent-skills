"""Shared calendar -- a real source of truth for dates so agents stop trusting
stale relative-time phrasing in old notes ("Copa 506 coming up" written months
ago). File-based JSON (like agent_reports/agent_mail) so it syncs to the server.

    calendar.add("2026-07-15", "Copa 506", type="tournament")
    calendar.upcoming(90)   # events from today forward
    calendar.past(30)       # recently finished
    calendar.remove(event_id)

Dates are ISO ``YYYY-MM-DD`` (also accepts "today"/"tomorrow").
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)


def _path() -> Path:
    return config.calendar_path


def _load() -> list[dict]:
    p = _path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save(events: list[dict]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_date(s: str) -> str:
    """Normalize to ISO. Accepts YYYY-MM-DD, 'today', 'tomorrow'."""
    s = s.strip().lower()
    if s == "today":
        return date.today().isoformat()
    if s == "tomorrow":
        return (date.today() + timedelta(days=1)).isoformat()
    return datetime.strptime(s, "%Y-%m-%d").date().isoformat()  # ValueError -> @skill failure


@skill
def add(when: str, title: str, type: str = "", notes: str = "") -> Result:
    """Add an event. ``when`` is YYYY-MM-DD (or today/tomorrow)."""
    iso = _parse_date(when)
    if not title.strip():
        return Result.failure("an event needs a title")
    events = _load()
    ev = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "date": iso, "title": title.strip(), "type": type.strip(),
        "notes": notes.strip(), "created": date.today().isoformat(),
    }
    events.append(ev)
    events.sort(key=lambda e: e["date"])
    _save(events)
    log.info("calendar add %s -- %s", iso, title)
    return Result.success(ev)


@skill
def remove(event_id: str) -> Result:
    events = _load()
    kept = [e for e in events if e["id"] != event_id]
    if len(kept) == len(events):
        return Result.failure(f"no event with id {event_id}")
    _save(kept)
    return Result.success({"removed": event_id})


@skill
def all_events() -> Result:
    return Result.success(sorted(_load(), key=lambda e: e["date"]))


@skill
def upcoming(days: int = 90) -> Result:
    """Events from today through ``days`` ahead, soonest first."""
    today = date.today().isoformat()
    end = (date.today() + timedelta(days=days)).isoformat()
    out = [e for e in sorted(_load(), key=lambda e: e["date"]) if today <= e["date"] <= end]
    return Result.success(out)


@skill
def past(days: int = 30) -> Result:
    """Events in the last ``days``, most recent first."""
    start = (date.today() - timedelta(days=days)).isoformat()
    today = date.today().isoformat()
    out = [e for e in sorted(_load(), key=lambda e: e["date"], reverse=True)
           if start <= e["date"] < today]
    return Result.success(out)


@skill
def on_date(when: str) -> Result:
    iso = _parse_date(when)
    return Result.success([e for e in _load() if e["date"] == iso])
