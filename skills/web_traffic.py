"""Traffic for the public web app (the chemistry calculator, LXC 103).

The calculator is the only service strangers can reach, so it is the only one
whose visitors are worth counting -- and the only one that gets scanned.

    summary(hours)   -> visits, unique visitors, status mix, top pages
    anomalies(hours) -> the "is anything off?" list
    report(hours)    -> a formatted block for Warden / the daily digest

Reads gunicorn's access log inside LXC 103 by way of the host, the same way
security_monitor reaches the auth journal.

The first field of each line is X-Forwarded-For, which Tailscale Funnel fills
with the real client address; %(h)s beside it is always the proxy (127.0.0.1)
and is ignored.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone

from skills.config import config
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)

ACCESS_LOG = "/var/log/chemcalc/access.log"
CONTAINER = "103"

# Paths nobody reaches by accident: someone is looking for a CMS, a shell, or
# credentials. One of these is a scanner; a burst of them is worth saying so.
_PROBE = re.compile(
    r"/(wp-admin|wp-login|xmlrpc|\.env|\.git|phpmyadmin|admin|administrator"
    r"|shell|cgi-bin|vendor|config\.|backup|\.aws|\.ssh|actuator|solr|boaform)",
    re.I)

# xff proxy [timestamp] GET /path HTTP/1.1 200 1234 5ms user agent
#
# The request and user-agent are matched with optional quotes. gunicorn's format
# string passes through a systemd unit and a shell, and whether the quotes
# survive that depends on how the unit was written -- a parser that only accepts
# one of the two silently matches nothing and reports a quiet day.
_LINE = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+'
    r'"?(?P<method>[A-Z]+)\s+(?P<path>\S+)\s+\S+?"?\s+'
    r'(?P<status>\d{3})\s+(?P<bytes>\S+)\s+(?P<ms>[\d.]+)ms\s*'
    r'"?(?P<ua>.*?)"?\s*$')

_STATIC = re.compile(r"^/static/")


def _host(cmd: str, timeout: float = 25) -> tuple[str, int]:
    full = ["ssh", "-i", config.server_ssh_key,
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10", f"root@{config.server_host}", cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip(), r.returncode
    except Exception as exc:
        return f"error: {exc}", 1


def _read_log(max_lines: int = 20000) -> tuple[list[str], str | None]:
    out, rc = _host(f"pct exec {CONTAINER} -- tail -n {max_lines} {ACCESS_LOG} 2>/dev/null")
    if rc != 0:
        return [], f"couldn't read the access log: {out[:150]}"
    return [l for l in out.splitlines() if l.strip()], None


def _parse(lines: list[str], hours: int) -> list[dict]:
    """Parsed hits inside the window, newest log format only."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    hits = []
    for line in lines:
        m = _LINE.match(line)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group("ts"), "%d/%b/%Y:%H:%M:%S %z")
        except ValueError:
            continue
        if ts < cutoff:
            continue
        d = m.groupdict()
        d["when"] = ts
        d["status"] = int(d["status"])
        hits.append(d)
    return hits


def _is_internal(ip: str) -> bool:
    """Your own devices: LAN, loopback, or Tailscale CGNAT."""
    if ip.startswith(("192.168.", "10.", "127.", "-")):
        return True
    if ip.startswith("100."):
        try:
            return 64 <= int(ip.split(".")[1]) <= 127
        except (IndexError, ValueError):
            return False
    return False


def summary(hours: int = 24) -> Result:
    """Traffic rollup for the last ``hours``."""
    lines, err = _read_log()
    if err:
        return Result.failure(err)
    hits = _parse(lines, hours)

    pages = [h for h in hits if not _STATIC.match(h["path"])]
    outside = [h for h in pages if not _is_internal(h["ip"])]
    visitors = {h["ip"] for h in pages if not _is_internal(h["ip"])}

    status = Counter(h["status"] // 100 for h in hits)
    slow = [h for h in hits if float(h["ms"]) > 2000]

    return Result.success({
        "hours": hours,
        "requests": len(hits),
        "page_requests": len(pages),
        "outside_requests": len(outside),
        "unique_visitors": len(visitors),
        "visitor_ips": sorted(visitors),
        "ok": status[2], "redirect": status[3],
        "client_errors": status[4], "server_errors": status[5],
        "top_paths": Counter(h["path"] for h in pages).most_common(5),
        "top_visitors": Counter(h["ip"] for h in outside).most_common(5),
        "slow_requests": len(slow),
    })


def anomalies(hours: int = 24) -> Result:
    """Anything that looks off. Empty list means a quiet day."""
    lines, err = _read_log()
    if err:
        return Result.failure(err)
    hits = _parse(lines, hours)
    found: list[str] = []

    probes = [h for h in hits if _PROBE.search(h["path"])]
    if probes:
        who = Counter(h["ip"] for h in probes).most_common(3)
        found.append(f"🔍 {len(probes)} scanner probes (wp-admin/.env/etc) from "
                     + ", ".join(f"{ip} ×{n}" for ip, n in who))

    errs = [h for h in hits if h["status"] >= 500]
    if errs:
        found.append(f"🔥 {len(errs)} server errors (5xx) — the app is failing on something: "
                     + ", ".join(sorted({h['path'] for h in errs})[:3]))

    # The input guards answering means someone sent something oversized.
    guarded = [h for h in hits if h["status"] in (400, 413)]
    if len(guarded) >= 20:
        found.append(f"🛡️ {len(guarded)} requests rejected by the size/length guards — "
                     "either a scripted client or someone probing the limits")

    outside = [h for h in hits if not _is_internal(h["ip"])]
    if outside:
        top_ip, top_n = Counter(h["ip"] for h in outside).most_common(1)[0]
        if top_n >= 300:
            found.append(f"📈 {top_ip} made {top_n} requests — well past normal browsing")

    slow = [h for h in hits if float(h["ms"]) > 5000]
    if slow:
        found.append(f"🐌 {len(slow)} requests took over 5s — possible CPU pressure")

    return Result.success({"anomalies": found, "clean": not found})


def report(hours: int = 24) -> Result:
    """One formatted block, for Warden's daily report and Iris' digest."""
    s = summary(hours)
    if not s.ok:
        return s
    d = s.data
    a = anomalies(hours)

    if d["outside_requests"] == 0:
        headline = "no outside visitors"
    else:
        headline = (f"{d['unique_visitors']} visitor"
                    f"{'s' if d['unique_visitors'] != 1 else ''}, "
                    f"{d['outside_requests']} requests")

    lines = [f"🌐 **Calculator traffic** ({hours}h): {headline}"]
    if d["requests"]:
        lines.append(f"   {d['ok']} ok · {d['client_errors']} client errors · "
                     f"{d['server_errors']} server errors")
        if d["top_paths"]:
            lines.append("   busiest: " + ", ".join(f"{p} ×{n}" for p, n in d["top_paths"][:3]))
    if a.ok and a.data["anomalies"]:
        lines += ["   " + x for x in a.data["anomalies"]]
    elif a.ok:
        lines.append("   nothing off")
    return Result.success({"text": "\n".join(lines), **d})
