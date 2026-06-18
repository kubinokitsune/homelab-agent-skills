"""Agent-to-agent mail -- the spider web's messaging layer.

Async and file-based (like agent_reports), so it syncs to the server later. An
agent drops a message in another agent's mailbox; the recipient polls its inbox,
acts on each message, and archives it.

    agent_mail.send("Axiom", "Forge", "new vault note", body="path/to/note")
    msgs = agent_mail.inbox("Axiom")     # [{from, to, subject, body, time, path}]
    agent_mail.archive(msg["path"])

Messages are JSON files under ``agent-mail/{recipient}/``; archived ones move to
``agent-mail/{recipient}/_archive/``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)


@skill
def send(to: str, frm: str, subject: str, body: str = "") -> Result:
    """Drop a message in another agent's mailbox."""
    folder = config.agent_mail_dir / to
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = folder / f"{stamp}__{frm}.json"
    path.write_text(
        json.dumps({
            "from": frm, "to": to, "subject": subject, "body": body,
            "time": datetime.now().isoformat(timespec="seconds"),
        }),
        encoding="utf-8",
    )
    log.info("mail %s -> %s: %s", frm, to, subject)
    return Result.success({"path": str(path)})


@skill
def inbox(agent: str) -> Result:
    """Return an agent's pending (un-archived) messages, oldest first."""
    folder = config.agent_mail_dir / agent
    if not folder.is_dir():
        return Result.success([])
    msgs = []
    for p in sorted(folder.glob("*.json")):  # glob skips the _archive subfolder
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        data["path"] = str(p)
        msgs.append(data)
    return Result.success(msgs)


@skill
def archive(message_path: str) -> Result:
    """Move a processed message into the recipient's _archive folder."""
    p = Path(message_path)
    if not p.exists():
        return Result.failure(f"message not found: {message_path}")
    archive_dir = p.parent / "_archive"
    archive_dir.mkdir(exist_ok=True)
    dest = archive_dir / p.name
    p.replace(dest)
    return Result.success({"archived": str(dest)})
