"""Activity feed -- a shared, append-only log of what the agents say and page.

Every proactive Discord post (``DiscordAgent.post``), every alert
(``notifications.notify`` / ``send_discord``) and every phone page
(``notifications.send_pushover``) is appended here as one JSON line. The hub
tails the file and streams new lines to open browsers, so agent alerts show up
there live -- without any agent knowing the hub exists. Agents and the hub share
a filesystem (LXC 100), and each line is one small append, so several agent
processes can write at once.

    events.record("page", "Print failing", severity="emergency", title="Mason")
    events.recent(50)          # newest first

Recording never raises: a feed problem must never be able to swallow an alert.
"""

from __future__ import annotations

import json
import os
import time

from skills.config import config

MAX_TEXT = 1800          # long posts (the morning digest) are clipped
_TRIM_AT = 2_000_000     # bytes -- past this the file is cut back to...
_KEEP = 1500             # ...the newest this many events

SEVERITIES = ("info", "warning", "critical", "emergency")


def path():
    return config.hub_events_path


def _agent_name() -> str:
    try:
        from skills.logging import _current_agent
        return _current_agent()
    except Exception:
        return ""


def record(kind: str, text: str, *, agent: str = "", severity: str = "info", title: str = "") -> None:
    """Append one event. ``kind``: post | alert | page. Never raises."""
    try:
        p = path()
        p.parent.mkdir(parents=True, exist_ok=True)
        ev = {
            "id": str(time.time_ns()),
            "ts": round(time.time(), 3),
            "kind": kind,
            "agent": agent or _agent_name(),
            "severity": severity if severity in SEVERITIES else "info",
            "title": title or "",
            "text": (text or "")[:MAX_TEXT],
        }
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        if p.stat().st_size > _TRIM_AT:
            _trim(p)
    except Exception:
        pass


def _trim(p) -> None:
    lines = p.read_text(encoding="utf-8").splitlines()[-_KEEP:]
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def recent(limit: int = 100) -> list[dict]:
    """The newest ``limit`` events, newest first."""
    p = path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out[::-1]


def post_severity(text: str) -> str:
    """How loud an agent's own channel post is: one that pings you or carries
    an alarm emoji wanted your attention; everything else is routine."""
    return "warning" if any(m in (text or "") for m in ("<@", "🚨", "🔴", "⚠️")) else "info"
