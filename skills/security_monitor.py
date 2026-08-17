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

import os
import re
import subprocess
from collections import Counter

from skills.config import config
from skills.logging import get_logger
from skills.result import Result

_log = get_logger("security_monitor")

# IPs Warden must NEVER ban -- the hard safety that stops it locking Pipe out.
# LAN + loopback + Tailscale (100.64.0.0/10) + any WARDEN_TRUSTED_IPS he sets.
_TRUSTED_ENV = [x.strip() for x in (os.getenv("WARDEN_TRUSTED_IPS") or "").split(",") if x.strip()]

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


def _is_trusted(ip: str) -> bool:
    """Never-ban list: LAN + loopback + Tailscale CGNAT (100.64.0.0/10) + configured IPs."""
    if _is_private(ip) or ip in _TRUSTED_ENV:
        return True
    if ip.startswith("100."):  # Tailscale 100.64.0.0/10
        try:
            return 64 <= int(ip.split(".")[1]) <= 127
        except (IndexError, ValueError):
            return False
    return False


def _banlist_path():
    return config.banlist_path


def _load_bans() -> set:
    p = _banlist_path()
    if not p.exists():
        return set()
    try:
        return {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}
    except Exception:
        return set()


def _save_bans(ips: set) -> None:
    p = _banlist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(sorted(ips)) + "\n")


def stored_bans() -> Result:
    """The IPs Warden has banned (source of truth, survives reboot)."""
    return Result.success(sorted(_load_bans()))


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
    """Drop all traffic from an IP at the host firewall (idempotent + persisted).
    REFUSES trusted IPs (LAN / Tailscale / you) so it can never lock you out."""
    if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ip):
        return Result.failure(f"'{ip}' isn't a valid IPv4 address")
    if _is_trusted(ip):
        return Result.failure(f"refusing to ban {ip} — it's a trusted LAN/Tailscale address")
    out, rc = _host(f"iptables -C INPUT -s {ip} -j DROP 2>/dev/null || iptables -A INPUT -s {ip} -j DROP")
    if rc != 0:
        return Result.failure(f"ban failed: {out}")
    bans = _load_bans()
    bans.add(ip)
    _save_bans(bans)
    return Result.success(f"banned {ip}")


def unban(ip: str) -> Result:
    """Remove the firewall drop for an IP and forget it."""
    if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ip):
        return Result.failure(f"'{ip}' isn't a valid IPv4 address")
    _host(f"iptables -D INPUT -s {ip} -j DROP 2>/dev/null")  # ignore "rule not found"
    bans = _load_bans()
    bans.discard(ip)
    _save_bans(bans)
    return Result.success(f"unbanned {ip}")


def reapply_bans() -> Result:
    """Re-add every stored ban at the firewall (iptables rules are lost on reboot).
    Warden calls this on startup so bans survive a host restart."""
    applied = 0
    bans = _load_bans()
    for ip in bans:
        if _is_trusted(ip):
            continue
        _, rc = _host(f"iptables -C INPUT -s {ip} -j DROP 2>/dev/null || iptables -A INPUT -s {ip} -j DROP")
        applied += 1 if rc == 0 else 0
    return Result.success({"reapplied": applied, "total": len(bans)})


def audit() -> Result:
    """Security-posture snapshot: SSH config, firewall, Tailscale, ban count."""
    out, _ = _host(
        "echo pwauth=$(sshd -T 2>/dev/null | awk '/^passwordauthentication/{print $2}'); "
        "echo rootlogin=$(sshd -T 2>/dev/null | awk '/^permitrootlogin/{print $2}'); "
        "echo dropcount=$(iptables -L INPUT -n 2>/dev/null | grep -c DROP); "
        "echo tailscale=$(command -v tailscale >/dev/null && (tailscale status >/dev/null 2>&1 && echo up || echo installed) || echo no)")
    f = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            f[k.strip()] = v.strip()
    f["banned"] = len(_load_bans())
    return Result.success(f)
