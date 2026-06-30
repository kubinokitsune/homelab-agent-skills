"""Security monitor -- Warden's eyes on who's knocking at the server.

Reads the host's sshd journal (Debian uses journald, not auth.log) and parses
auth events: failed passwords, invalid users, and successful logins. The agents
(Hermes/Mason/Warden) SSH to the host constantly with a key from the LAN, so
those are filtered out as "internal" -- Warden only cares about the *unexpected*:
a successful login that isn't LAN+publickey, or a brute-force burst.

    summary()        -> counts, top offender IPs, suspicious logins
    sessions()       -> who's logged in right now
    listening()      -> open listening ports
    ban("1.2.3.4")   -> drop that IP at the host firewall

Reaches the host over SSH (config.server_ssh_key), like server_monitor.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter

from skills.config import config
from skills.logging import get_logger
from skills.result import Result

_log = get_logger("security_monitor")

_ACCEPT = re.compile(r"Accepted (\w+) for (\S+) from (\d+\.\d+\.\d+\.\d+)")
_FAIL = re.compile(r"Failed password for (?:invalid user )?(\S+) from (\d+\.\d+\.\d+\.\d+)")
_INVALID = re.compile(r"Invalid user (\S+) from (\d+\.\d+\.\d+\.\d+)")


def _host(cmd: str, timeout: float = 20) -> tuple[str, int]:
    full = ["ssh", "-i", config.server_ssh_key,
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10", f"root@{config.server_host}", cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip(), r.returncode
    except Exception as exc:
        return f"error: {exc}", 1


def _is_private(ip: str) -> bool:
    return (ip.startswith("192.168.") or ip.startswith("10.")
            or any(ip.startswith(f"172.{i}.") for i in range(16, 32)) or ip.startswith("127."))


def auth_events(since: str = "-24h") -> Result:
    """Parsed sshd auth events in the window: accepted / failed / invalid."""
    out, rc = _host(f"journalctl -u ssh --since '{since}' --no-pager 2>/dev/null")
    if rc != 0:
        return Result.failure(f"couldn't read auth journal: {out}")
    accepted, failed, invalid = [], [], []
    for line in out.splitlines():
        if (m := _ACCEPT.search(line)):
            accepted.append({"method": m.group(1), "user": m.group(2), "ip": m.group(3)})
        elif (m := _FAIL.search(line)):
            failed.append({"user": m.group(1), "ip": m.group(2)})
        elif (m := _INVALID.search(line)):
            invalid.append({"user": m.group(1), "ip": m.group(2)})
    return Result.success({"accepted": accepted, "failed": failed, "invalid": invalid})


def summary(since: str = "-24h") -> Result:
    """Security rollup: counts, top offender IPs, and any *suspicious* logins."""
    ev = auth_events(since)
    if not ev.ok:
        return ev
    d = ev.data
    fail_by_ip = Counter(f["ip"] for f in d["failed"])
    # Suspicious = a successful login that ISN'T the agents' LAN+publickey access.
    suspicious = [a for a in d["accepted"]
                  if not (_is_private(a["ip"]) and a["method"] == "publickey")]
    internal = len(d["accepted"]) - len(suspicious)
    return Result.success({
        "failed": len(d["failed"]),
        "invalid": len(d["invalid"]),
        "accepted": len(d["accepted"]),
        "internal_logins": internal,
        "suspicious_logins": suspicious,
        "top_offenders": fail_by_ip.most_common(5),
    })


def sessions() -> Result:
    """Active login sessions (who)."""
    out, rc = _host("who")
    if rc != 0:
        return Result.failure(f"who failed: {out}")
    rows = []
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 5:
            rows.append({"user": f[0], "when": " ".join(f[2:4]), "from": f[-1].strip("()")})
    return Result.success(rows)


def listening() -> Result:
    """Listening TCP ports on the host (the external surface)."""
    out, rc = _host("ss -tlnH | awk '{print $4}' | sort -u")
    if rc != 0:
        return Result.failure(f"ss failed: {out}")
    return Result.success([p for p in out.splitlines() if p])


def ban(ip: str) -> Result:
    """Drop all traffic from an IP at the host firewall (idempotent)."""
    if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ip):
        return Result.failure(f"'{ip}' isn't a valid IPv4 address")
    out, rc = _host(f"iptables -C INPUT -s {ip} -j DROP 2>/dev/null || iptables -A INPUT -s {ip} -j DROP")
    if rc != 0:
        return Result.failure(f"ban failed: {out}")
    return Result.success(f"banned {ip}")
