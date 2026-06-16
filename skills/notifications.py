"""Unified alerting -- the one way any agent talks to you.

Two layers:

* **Primitives** -- ``send_discord`` and ``send_pushover``. Explicit, each
  returns a ``Result``.
* **Policy** -- ``notify(message, severity=...)`` routes by severity so agents
  don't hardcode channels. The routing table (from the master doc):

      INFO      -> Discord, the agent's own channel        (routine signal)
      WARNING   -> Discord #warnings                        (needs attention)
      CRITICAL  -> Discord #critical + Pushover priority 1  (record + phone, bypasses DND)

Notes baked in from setup testing:

* Discord sits behind Cloudflare, which 403s the default ``Python-urllib``
  user-agent. Every request here sends a real ``User-Agent`` -- non-negotiable.
* The webhook's ``username`` is set to the agent name, so a glance at #warnings
  tells you whether it was Mason or Warden talking.
"""

from __future__ import annotations

import enum
import json
import urllib.error
import urllib.parse
import urllib.request

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)

# Identifies us to Discord/Cloudflare so requests aren't 403'd as a bot-with-no-UA.
_USER_AGENT = "homelab-agent/0.1 (+https://github.com/kubinokitsune/homelab-agent-skills)"
_PUSHOVER_API = "https://api.pushover.net/1/messages.json"
_TIMEOUT = 15


class Severity(enum.Enum):
    """How loud an alert should be -- drives where ``notify`` sends it."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# -- low-level HTTP helpers ------------------------------------------------
# These return (status_code, body) and swallow HTTPError so a 4xx/5xx becomes a
# clean Result.failure upstream rather than an exception. Genuine network
# failures (no DNS, no route) still raise and are caught by @skill.

def _post(url: str, data: bytes, content_type: str) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": content_type, "User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")


# -- primitives ------------------------------------------------------------

@skill
def send_discord(channel: str, message: str) -> Result:
    """Post a plain message to a Discord channel's webhook.

    ``channel`` is the logical name ("warnings"), resolved to its webhook via
    config. The sending agent's name appears as the Discord username.
    """
    url = config.discord_webhook(channel)
    payload = json.dumps({"content": message, "username": config.agent_name}).encode()
    code, body = _post(url, payload, "application/json")
    if code in (200, 204):
        log.info("discord -> #%s ok", channel)
        return Result.success({"channel": channel, "status": code})
    return Result.failure(f"Discord #{channel} returned {code}: {body[:200]}")


@skill
def send_pushover(message: str, title: str | None = None, priority: int = 0) -> Result:
    """Send a Pushover push notification.

    ``priority=1`` is high priority and bypasses the phone's quiet hours -- use
    it for genuinely critical alerts. Default 0 is normal.
    """
    fields = {
        "token": config.pushover_token,
        "user": config.pushover_user,
        "message": message,
        "priority": str(priority),
    }
    if title:
        fields["title"] = title
    code, body = _post(_PUSHOVER_API, urllib.parse.urlencode(fields).encode(),
                       "application/x-www-form-urlencoded")
    if code == 200 and '"status":1' in body.replace(" ", ""):
        log.info("pushover ok (priority=%s)", priority)
        return Result.success({"status": code})
    return Result.failure(f"Pushover returned {code}: {body[:200]}")


# -- policy ----------------------------------------------------------------

def _format(message: str, title: str | None) -> str:
    """Bold the title above the message for Discord, if a title is given."""
    return f"**{title}**\n{message}" if title else message


def notify(
    message: str,
    severity: Severity = Severity.INFO,
    channel: str | None = None,
    title: str | None = None,
) -> Result:
    """Send an alert, routed by severity. The high-level entry point.

    INFO     -> Discord, ``channel`` (defaults to the agent's own channel).
    WARNING  -> Discord #warnings.
    CRITICAL -> Discord #critical AND Pushover (priority 1, bypasses DND).

    Returns a single ``Result``. For CRITICAL, which sends to two places, the
    result is successful only if both sends succeed; otherwise it carries the
    combined reason.
    """
    if severity is Severity.INFO:
        target = channel or config.agent_name.lower()
        return send_discord(target, _format(message, title))

    if severity is Severity.WARNING:
        return send_discord("warnings", _format(message, title))

    # CRITICAL: log to Discord for the record, and buzz the phone past DND.
    discord_result = send_discord("critical", _format(message, title))
    pushover_result = send_pushover(message, title=title or "Critical alert", priority=1)

    if discord_result.ok and pushover_result.ok:
        return Result.success({"discord": discord_result.data, "pushover": pushover_result.data})

    reasons = [r.error for r in (discord_result, pushover_result) if not r.ok]
    return Result.failure("CRITICAL alert partially failed: " + "; ".join(reasons))
