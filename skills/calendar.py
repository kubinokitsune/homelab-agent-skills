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


def _norm_time(t: str) -> str:
    """Normalize an HH:MM string, or '' if blank/invalid (never raises)."""
    t = (t or "").strip()
    if not t:
        return ""
    try:
        return datetime.strptime(t, "%H:%M").strftime("%H:%M")
    except ValueError:
        return ""


@skill
def add(when: str, title: str, type: str = "", notes: str = "",
        start: str = "", end: str = "") -> Result:
    """Add an event. ``when`` is YYYY-MM-DD (or today/tomorrow). ``start``/``end``
    are optional ``HH:MM`` clock times -- give them and the event occupies real
    hours (so Kairos can block time and spot conflicts); omit them for an all-day
    event. Events written before times existed stay valid (no start/end)."""
    iso = _parse_date(when)
    if not title.strip():
        return Result.failure("an event needs a title")
    s, e = _norm_time(start), _norm_time(end)
    if s and e and e <= s:
        return Result.failure("the end time must be after the start time")
    events = _load()
    # Unique id: a timestamp to the second used to collide when two events were
    # added within the same second (then removing one removed both).
    taken = {e["id"] for e in events}
    eid = datetime.now().strftime("%Y%m%d%H%M%S")
    n = 1
    while eid in taken:
        eid = f"{datetime.now():%Y%m%d%H%M%S}-{n}"
        n += 1
    ev = {
        "id": eid,
        "date": iso, "title": title.strip(), "type": type.strip(),
        "start": s, "end": e,
        "notes": notes.strip(), "created": date.today().isoformat(),
    }
    events.append(ev)
    events.sort(key=lambda e: (e["date"], e.get("start") or "99:99"))
    _save(events)
    log.info("calendar add %s %s-%s -- %s", iso, s or "--", e or "--", title)
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
    return Result.success(sorted(_load(), key=lambda e: (e["date"], e.get("start") or "99:99")))


@skill
def upcoming(days: int = 90) -> Result:
    """Events from today through ``days`` ahead, soonest first."""
    today = date.today().isoformat()
    end = (date.today() + timedelta(days=days)).isoformat()
    out = [e for e in sorted(_load(), key=lambda e: (e["date"], e.get("start") or "99:99")) if today <= e["date"] <= end]
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
