"""Camera modes -- day / low-light / night V4L2 profiles for the print cam.

The webcam lives on the Proxmox host (ustreamer streams it); Mason reaches the
host over SSH (config.server_ssh_key) to set V4L2 controls. Applying a mode is
live *and* persisted: it rewrites /usr/local/bin/camera-mode.sh, which
camera-stream.service re-runs on every (re)start, so the chosen mode survives
reboots.

    camera.apply("day")        # bright room -- minimal gain, let auto-exposure work
    camera.apply("lowlight")   # dim/evening -- moderate boost, keeps white-PLA detail
    camera.apply("night")      # dark room -- aggressive gain/gamma for max visibility
    camera.current_mode()      # -> 'day' / 'lowlight' / 'night' / 'unknown'
"""

from __future__ import annotations

import shlex
import subprocess

from skills.config import config
from skills.logging import get_logger
from skills.result import Result

_log = get_logger("camera")

# All keep auto_exposure=3 (aperture priority) so exposure adapts; the gradient is
# gain/gamma/brightness. day = no boost (fixes daylight over-exposure) -> night = max.
PROFILES = {
    "day":      "auto_exposure=3,gain=0,gamma=90,brightness=-12,contrast=36,sharpness=3,backlight_compensation=0",
    "lowlight": "auto_exposure=3,gain=26,gamma=120,brightness=0,contrast=40,sharpness=4,backlight_compensation=2",
    "night":    "auto_exposure=3,gain=62,gamma=165,brightness=14,contrast=40,sharpness=4,backlight_compensation=0",
}


def _ssh(remote: str, timeout: float = 15) -> tuple[str, int]:
    cmd = ["ssh", "-i", config.server_ssh_key,
           "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-o", "ConnectTimeout=10", f"root@{config.server_host}", remote]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip(), r.returncode
    except Exception as exc:
        return f"error: {exc}", 1


def apply(profile: str) -> Result:
    """Apply a camera mode live and persist it for restarts."""
    profile = profile.lower()
    if profile not in PROFILES:
        return Result.failure(f"unknown mode '{profile}' — use day / lowlight / night")
    ctrls = PROFILES[profile]
    out, rc = _ssh(f"v4l2-ctl -d /dev/video0 --set-ctrl {ctrls}")
    if rc != 0:
        return Result.failure(f"couldn't apply: {out}")
    # persist: regenerate the startup script + record the active mode
    script = f"#!/bin/sh\n# Camera mode: {profile}\nsleep 2\nv4l2-ctl -d /dev/video0 --set-ctrl {ctrls}\n"
    _ssh("printf '%s' " + shlex.quote(script) +
         " > /usr/local/bin/camera-mode.sh && chmod +x /usr/local/bin/camera-mode.sh"
         f" && echo {profile} > /usr/local/bin/.camera-mode")
    return Result.success(profile)


def current_mode() -> Result:
    out, rc = _ssh("cat /usr/local/bin/.camera-mode 2>/dev/null")
    return Result.success(out if rc == 0 and out else "unknown")


# Modes ordered by how much they brighten the image (day = least, night = most).
MODES_ORDER = ["day", "lowlight", "night"]

# Empirical per-mode brightening factor relative to 'day' (gain/gamma effect),
# measured live: day reads ~25, lowlight ~68, night ~178 in the same dark room.
# These are properties of the *mode*, not the room, so we can use them to back
# out the ambient light from a reading taken in ANY mode -> a consistent estimate.
_MODE_FACTOR = {"day": 1.0, "lowlight": 2.7, "night": 7.1}
_TARGET = 110.0      # ideal average frame brightness (0-255): lit but not blown out
_SWITCH_MARGIN = 18  # only change modes if it improves exposure by at least this much


def frame_brightness(timeout: float = 8) -> Result:
    """Average brightness (0-255) of the current camera frame."""
    try:
        import io
        import urllib.request
        import numpy as np
        from PIL import Image
        data = urllib.request.urlopen(config.camera_snapshot_url, timeout=timeout).read()
        im = Image.open(io.BytesIO(data)).convert("L")
        return Result.success(float(np.asarray(im).mean()))
    except Exception as exc:
        return Result.failure(f"brightness read failed: {exc}")


def auto_step() -> Result:
    """Pick the camera mode that best matches the current room light.

    The old version compared the raw frame brightness to fixed thresholds, but
    each mode brightens the image so differently (day~25 / lowlight~68 / night~178
    in the same room) that every mode read "in range" and it never switched.

    Instead: estimate the ambient light independent of the current mode (divide
    the reading by that mode's brightening factor), predict what each mode WOULD
    produce for that ambient, and choose the one closest to a well-exposed target.
    A hysteresis margin stops it flapping between two modes at a boundary. This
    converges to one correct mode and tracks the light dawn-to-dusk on its own.
    """
    b = frame_brightness()
    if not b.ok:
        return b
    cur = current_mode().data
    if cur not in _MODE_FACTOR:
        cur = "lowlight"
    reading = b.data
    ambient = reading / _MODE_FACTOR[cur]  # day-equivalent ambient light

    def dist(mode: str) -> float:  # how far this mode's output is from ideal
        predicted = min(255.0, ambient * _MODE_FACTOR[mode])
        return abs(predicted - _TARGET)

    best = min(MODES_ORDER, key=dist)
    # Only switch if 'best' beats staying put by a clear margin (anti-flap).
    new = best if (best != cur and dist(cur) - dist(best) >= _SWITCH_MARGIN) else cur
    if new != cur:
        apply(new)
    return Result.success({"brightness": round(reading, 1), "ambient": round(ambient, 1),
                           "from": cur, "to": new, "changed": new != cur})
