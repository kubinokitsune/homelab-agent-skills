"""Server monitor -- Hermes's senses and hands on the homelab.

Hermes runs in the agents container (LXC 100) but manages the whole box: the
Proxmox host, the containers, and every service. Host/container actions go over
SSH to the host (key in config.server_ssh_key); the agents and the Docker
services are local to LXC 100.

    h = server_monitor.host_health()        # cpu/ram/disk/temp/uptime
    s = server_monitor.services()           # {name: 'active'|'dead'|...}
    server_monitor.restart_service("agent-scout")

Synchronous (subprocess); call from the async agent via asyncio.to_thread.
"""

from __future__ import annotations

import subprocess

from skills.config import config
from skills.logging import get_logger
from skills.result import Result

_log = get_logger("server_monitor")

# --- what Hermes watches, and where each service lives --------------------
AGENTS = ["agent-forge", "agent-scout", "agent-axiom", "agent-iris",
          "agent-kairos", "agent-chiron", "agent-codex", "agent-mason",
          "agent-hermes", "agent-warden"]
LXC101_SERVICES = ["klipper", "moonraker", "nginx"]   # printer container
DOCKER_SERVICES = ["ollama", "qdrant"]                # docker in LXC 100
HOST_SERVICES = ["camera-stream"]                     # systemd on the host


def _run(cmd: list[str], timeout: float = 15) -> tuple[str, int]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip(), r.returncode
    except Exception as exc:
        return f"error: {exc}", 1


def _ssh(remote: str, timeout: float = 15) -> tuple[str, int]:
    """Run a command on the Proxmox host over SSH."""
    return _run(["ssh", "-i", config.server_ssh_key,
                 "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                 "-o", "ConnectTimeout=10", f"root@{config.server_host}", remote], timeout)


def _local(sh: str, timeout: float = 15) -> tuple[str, int]:
    return _run(["bash", "-lc", sh], timeout)


# --- host health ---------------------------------------------------------

def host_health() -> Result:
    """CPU load, RAM, disk, CPU temp, uptime of the Proxmox host."""
    out, rc = _ssh(
        "cat /proc/loadavg; echo ===; free -b | awk '/Mem:/{print $2,$3}'; echo ===; "
        "df -B1 / | tail -1 | awk '{print $2,$3,$5}'; echo ===; "
        "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1; echo ===; "
        "uptime -p")
    if rc != 0:
        return Result.failure(f"host unreachable: {out}")
    try:
        load_s, mem_s, disk_s, temp_s, up_s = [p.strip() for p in out.split("===")]
        load = load_s.split()[:3]
        mtot, mused = (int(x) for x in mem_s.split())
        dtot, dused, dpct = disk_s.split()
        temp_c = round(int(temp_s) / 1000, 1) if temp_s.isdigit() else None
        return Result.success({
            "load": load,
            "ram_used_gb": round(mused / 1e9, 1), "ram_total_gb": round(mtot / 1e9, 1),
            "ram_pct": round(mused / mtot * 100),
            "disk_used_gb": round(int(dused) / 1e9, 1), "disk_total_gb": round(int(dtot) / 1e9, 1),
            "disk_pct": dpct,
            "cpu_temp": temp_c, "uptime": up_s,
        })
    except Exception as exc:
        return Result.failure(f"couldn't parse host health: {exc}")


def free_disk() -> Result:
    """Reclaim host disk safely: vacuum old journals + clear the apt cache. Returns
    MB freed and the disk % afterward. No user data is touched."""
    def _used() -> int | None:
        out, _ = _ssh("df -B1 / | tail -1 | awk '{print $3}'")
        try:
            return int(out.strip())
        except ValueError:
            return None
    before = _used()
    _ssh("journalctl --vacuum-time=3d >/dev/null 2>&1; "
         "apt-get -y clean >/dev/null 2>&1; "
         "pct exec 100 -- journalctl --vacuum-time=3d >/dev/null 2>&1", timeout=90)
    after = _used()
    pct, _ = _ssh("df / | tail -1 | awk '{print $5}'")
    freed = round((before - after) / 1e6) if (before is not None and after is not None) else 0
    return Result.success({"freed_mb": max(0, freed), "after_pct": pct.strip() or "?"})


def top_memory(n: int = 4) -> Result:
    """Top memory-consuming processes on the host (container procs show too)."""
    out, rc = _ssh(f"ps -eo pmem,comm --sort=-pmem --no-headers | head -{int(n)}")
    if rc != 0:
        return Result.failure(f"ps failed: {out}")
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            try:
                procs.append({"name": parts[1].strip(), "pct": float(parts[0])})
            except ValueError:
                continue
    return Result.success(procs)


def hardware() -> Result:
    """Real hardware specs read live from the host -- so Hermes never guesses."""
    def q(cmd: str) -> str:
        out, rc = _ssh(cmd)
        return out.strip() if rc == 0 else ""
    return Result.success({
        "model": q("dmidecode -s system-product-name 2>/dev/null") or "unknown",
        "cpu": q("lscpu | grep 'Model name' | cut -d: -f2- | sed 's/^ *//'"),
        "cores": q("nproc"),
        "threads": q("lscpu | grep -m1 '^CPU(s):' | awk '{print $2}'"),
        "ram_total": q("free -h | awk '/^Mem:/{print $2}'"),
        "ram_kind": q("dmidecode -t memory 2>/dev/null | grep -m1 'Type: DDR' | cut -d: -f2 | sed 's/^ *//'"),
        "ram_speed": q("dmidecode -t memory 2>/dev/null | grep -m1 'Configured Memory Speed' | cut -d: -f2 | sed 's/^ *//'"),
        "ram_max": q("dmidecode -t memory 2>/dev/null | grep 'Maximum Capacity' | cut -d: -f2 | sed 's/^ *//'"),
        "slots": q("dmidecode -t memory 2>/dev/null | grep 'Number Of Devices' | cut -d: -f2 | sed 's/^ *//'"),
        "disk": q("lsblk -d -o SIZE,MODEL,ROTA | awk 'NR==2{print $1, $2, ($3==0?\"(SSD)\":\"(HDD)\")}'"),
    })


def containers() -> Result:
    """LXC containers and their status (from pct list)."""
    out, rc = _ssh("pct list | tail -n +2")
    if rc != 0:
        return Result.failure(f"pct list failed: {out}")
    cts = []
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 3:
            cts.append({"id": f[0], "status": f[1], "name": f[-1]})
    return Result.success(cts)


# --- services ------------------------------------------------------------

def services() -> Result:
    """Health of every watched service: {name: 'active'|'dead'|'unknown'}."""
    status: dict[str, str] = {}

    # agents (local to LXC 100) -- one call, one line each
    out, _ = _local("systemctl is-active " + " ".join(AGENTS))
    for name, st in zip(AGENTS, out.splitlines()):
        status[name] = "active" if st == "active" else "dead"

    # printer container services (host -> pct exec 101)
    out, rc = _ssh("pct exec 101 -- systemctl is-active " + " ".join(LXC101_SERVICES))
    states = out.splitlines() if rc == 0 or out else []
    for name, st in zip(LXC101_SERVICES, states):
        status[name] = "active" if st == "active" else "dead"

    # docker services (local)
    out, _ = _local("docker ps --format '{{.Names}}'")
    running = set(out.splitlines())
    for name in DOCKER_SERVICES:
        status[name] = "active" if name in running else "dead"

    # host services
    for name in HOST_SERVICES:
        out, _ = _ssh(f"systemctl is-active {name}")
        status[name] = "active" if out == "active" else "dead"

    return Result.success(status)


# Where to restart each kind of service.
def restart_service(name: str) -> Result:
    """Restart a watched service wherever it lives."""
    if name in AGENTS:
        out, rc = _local(f"systemctl restart {name}")
    elif name in LXC101_SERVICES:
        out, rc = _ssh(f"pct exec 101 -- systemctl restart {name}")
    elif name in DOCKER_SERVICES:
        out, rc = _local(f"docker restart {name}")
    elif name in HOST_SERVICES:
        out, rc = _ssh(f"systemctl restart {name}")
    else:
        return Result.failure(f"unknown service '{name}'")
    if rc != 0:
        return Result.failure(f"restart failed: {out}")
    return Result.success(f"restarted {name}")
