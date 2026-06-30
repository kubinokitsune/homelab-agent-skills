"""Moonraker client -- Mason's hands on the Ender 3 S1 Pro.

Mason runs in the agents container (LXC 100); Klipper/Moonraker run in the
dedicated printer container (LXC 101) at ``MOONRAKER_URL``. This skill is the
bridge: a thin, dependency-free (stdlib ``urllib``) HTTP client over Moonraker's
REST API, returning the library's standard ``Result`` so an unattended agent can
branch on failure instead of crashing at 3am mid-print.

Everything here is synchronous; call it from an async agent via
``asyncio.to_thread(moonraker.status)`` exactly like the other skills.

    st = moonraker.status()
    if st.ok:
        print(st.data["state"], st.data["hotend"]["temp"])

Z-offset note: the *live* first-layer adjustment Mason makes is the gcode offset
(babystep), ``gcode_move.homing_origin[2]``. ``babystep_z(+0.02)`` nudges the
nozzle away from the bed -- the fix for first-layer lines that look split down
the middle (nozzle too close / over-squished). ``save_zoffset()`` bakes the live
offset into the saved probe value so it survives a restart.
"""

from __future__ import annotations

import json
import posixpath
import urllib.parse
import urllib.request

from skills.config import config
from skills.logging import get_logger
from skills.result import Result

_log = get_logger("moonraker")

# Printer objects we care about in a status poll.
_QUERY_OBJECTS = (
    "print_stats", "heater_bed", "extruder", "gcode_move",
    "toolhead", "virtual_sdcard", "display_status", "webhooks", "fan",
)


def _url(path: str) -> str:
    return config.moonraker_url.rstrip("/") + path


def _get(path: str, timeout: float = 5.0) -> Result:
    try:
        with urllib.request.urlopen(_url(path), timeout=timeout) as resp:
            return Result.success(json.loads(resp.read().decode("utf-8")))
    except Exception as exc:  # network, timeout, json -- all "printer unreachable"
        return Result.failure(f"Moonraker GET {path} failed: {exc}")


def _post(path: str, timeout: float = 10.0) -> Result:
    try:
        req = urllib.request.Request(_url(path), data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return Result.success(json.loads(body) if body else {})
    except Exception as exc:
        return Result.failure(f"Moonraker POST {path} failed: {exc}")


# -- read -----------------------------------------------------------------

def status() -> Result:
    """A tidy snapshot of the printer: state, temps, print progress, z-offset."""
    objs = "&".join(urllib.parse.quote(o) for o in _QUERY_OBJECTS)
    res = _get(f"/printer/objects/query?{objs}")
    if not res.ok:
        return res
    try:
        s = res.data["result"]["status"]
    except (KeyError, TypeError):
        return Result.failure("Moonraker returned an unexpected status shape")

    ps = s.get("print_stats", {})
    vsd = s.get("virtual_sdcard", {})
    gm = s.get("gcode_move", {})
    wh = s.get("webhooks", {})
    homing_origin = gm.get("homing_origin") or [0, 0, 0, 0]

    return Result.success({
        "state": ps.get("state", wh.get("state", "unknown")),  # standby/printing/paused/complete/error
        "klippy": wh.get("state", "unknown"),                   # ready/startup/shutdown/error
        "klippy_message": wh.get("state_message", ""),
        "filename": ps.get("filename", ""),
        "progress": round((vsd.get("progress") or 0) * 100, 1),  # %
        "print_time": ps.get("print_duration", 0),               # elapsed seconds
        "filament_mm": ps.get("filament_used", 0) or 0,
        "layer": (ps.get("info") or {}).get("current_layer"),
        "total_layers": (ps.get("info") or {}).get("total_layer"),
        "hotend": {
            "temp": round(s.get("extruder", {}).get("temperature", 0), 1),
            "target": round(s.get("extruder", {}).get("target", 0), 1),
        },
        "bed": {
            "temp": round(s.get("heater_bed", {}).get("temperature", 0), 1),
            "target": round(s.get("heater_bed", {}).get("target", 0), 1),
        },
        "z_offset": round(homing_origin[2], 3),                 # live babystep offset
        "speed_factor": round((gm.get("speed_factor") or 1) * 100),
        "position": [round(p, 2) for p in (s.get("toolhead", {}).get("position") or [0, 0, 0, 0])],
        # live, runtime-tunable settings (differ from the slicer's baked values)
        "flow": round((gm.get("extrude_factor") or 1) * 100),    # extrusion multiplier %
        "fan": round((s.get("fan", {}).get("speed") or 0) * 100),  # part-cooling fan %
        "pressure_advance": round(s.get("extruder", {}).get("pressure_advance", 0) or 0, 4),
        "max_accel": round(s.get("toolhead", {}).get("max_accel", 0) or 0),
        "max_velocity": round(s.get("toolhead", {}).get("max_velocity", 0) or 0),
        "square_corner_velocity": round(s.get("toolhead", {}).get("square_corner_velocity", 0) or 0, 1),
    })


def list_files(limit: int = 15) -> Result:
    """Recent gcode files on the printer, newest first."""
    res = _get("/server/files/list?root=gcodes")
    if not res.ok:
        return res
    files = res.data.get("result", [])
    files.sort(key=lambda f: f.get("modified", 0), reverse=True)
    return Result.success([f["path"] for f in files[:limit]])


def file_metadata(filename: str) -> Result:
    """Slicer metadata for a gcode file: estimated_time, layer_count, filament.

    Parsed by Moonraker at upload, so it's the best source for an accurate ETA.
    """
    res = _get(f"/server/files/metadata?filename={urllib.parse.quote(filename)}")
    if not res.ok:
        return res
    m = res.data.get("result", {})
    return Result.success({
        "estimated_time": m.get("estimated_time"),   # seconds (slicer estimate)
        "layer_count": m.get("layer_count"),
        "filament_total": m.get("filament_total"),    # mm
        "filament_weight": m.get("filament_weight_total"),  # grams
        "filament_type": m.get("filament_type"),
        "object_height": m.get("object_height"),      # mm
        "layer_height": m.get("layer_height"),         # mm
        "first_layer_height": m.get("first_layer_height"),
        "nozzle_diameter": m.get("nozzle_diameter"),
        "slicer": m.get("slicer"),
        "thumbnails": m.get("thumbnails") or [],       # [{width,height,relative_path}]
    })


def fetch_thumbnail(filename: str, thumbnails: list | None = None) -> Result:
    """Fetch the largest slicer-embedded preview image (PNG bytes) for a gcode.

    Slicers (Orca/Prusa/Cura) bake a render of the model into the gcode; Moonraker
    serves it. Pass the metadata's ``thumbnails`` list to avoid a second metadata
    fetch, or omit it and we'll look it up.
    """
    if thumbnails is None:
        meta = file_metadata(filename)
        if not meta.ok:
            return Result.failure(meta.error)
        thumbnails = meta.data.get("thumbnails") or []
    if not thumbnails:
        return Result.failure("no embedded thumbnail in this gcode")
    best = max(thumbnails, key=lambda t: (t.get("width", 0) * t.get("height", 0)))
    rel = best.get("relative_path") or best.get("thumbnail_path")
    if not rel:
        return Result.failure("thumbnail metadata missing a path")
    # relative_path is relative to the gcode's own directory.
    gpath = posixpath.normpath(posixpath.join(posixpath.dirname(filename), rel))
    url = "/server/files/gcodes/" + urllib.parse.quote(gpath, safe="/")
    try:
        with urllib.request.urlopen(_url(url), timeout=8) as resp:
            return Result.success(resp.read())
    except Exception as exc:
        return Result.failure(f"thumbnail fetch failed: {exc}")


# -- write: motion / temps ------------------------------------------------

def gcode(script: str) -> Result:
    """Run a raw gcode/Klipper command (e.g. ``G28``, ``QUERY_PROBE``)."""
    return _post(f"/printer/gcode/script?script={urllib.parse.quote(script)}")


def set_temp(heater: str, target: float) -> Result:
    """Set a heater target. ``heater`` is 'bed' or 'hotend'/'extruder'."""
    h = heater.lower()
    if h in ("bed", "heater_bed"):
        return gcode(f"M140 S{target:g}")
    if h in ("hotend", "extruder", "nozzle"):
        return gcode(f"M104 S{target:g}")
    return Result.failure(f"Unknown heater '{heater}' (use 'bed' or 'hotend')")


def home(axes: str = "") -> Result:
    """Home axes. ``home()`` homes all; ``home('z')`` homes one."""
    return gcode("G28" + (f" {axes.upper()}" if axes else ""))


def cooldown() -> Result:
    """Turn off both heaters."""
    return gcode("TURN_OFF_HEATERS")


# -- write: live tuning (safe to change mid-print) ------------------------

def set_speed_factor(pct: float) -> Result:
    """Set print-speed override (M220), percent of programmed feedrate."""
    return gcode(f"M220 S{max(1, round(pct))}")


def set_flow(pct: float) -> Result:
    """Set extrusion-flow override (M221), percent."""
    return gcode(f"M221 S{max(1, round(pct))}")


def set_fan(pct: float) -> Result:
    """Set part-cooling fan, percent (0 = off)."""
    pct = max(0, min(100, pct))
    return gcode("M107" if pct == 0 else f"M106 S{round(pct * 255 / 100)}")


def set_pressure_advance(value: float) -> Result:
    """Set pressure advance for the active extruder (live tuning)."""
    return gcode(f"SET_PRESSURE_ADVANCE ADVANCE={value:.4f}")


# -- write: print control -------------------------------------------------

def upload_gcode(filename: str, data: bytes, start: bool = False) -> Result:
    """Upload a gcode file to the printer (root=gcodes), optionally auto-starting it.

    Builds a multipart/form-data body by hand (stdlib only). ``start=True`` tells
    Moonraker to begin printing it right after upload.
    """
    boundary = "----MasonBoundary7MA4YWxkTrZu0gW"

    def _field(name: str, value: str) -> bytes:
        return (f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="{name}"\r\n\r\n{value}\r\n').encode()

    body = _field("root", "gcodes") + _field("print", "true" if start else "false")
    body += (f"--{boundary}\r\nContent-Disposition: form-data; "
             f'name="file"; filename="{filename}"\r\n'
             "Content-Type: application/octet-stream\r\n\r\n").encode()
    body += data + b"\r\n" + f"--{boundary}--\r\n".encode()
    try:
        req = urllib.request.Request(
            _url("/server/files/upload"), data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return Result.success(json.loads(resp.read().decode("utf-8")))
    except Exception as exc:
        return Result.failure(f"upload failed: {exc}")


def start_print(filename: str) -> Result:
    return _post(f"/printer/print/start?filename={urllib.parse.quote(filename)}")


def pause() -> Result:
    return _post("/printer/print/pause")


def resume() -> Result:
    return _post("/printer/print/resume")


def cancel() -> Result:
    return _post("/printer/print/cancel")


def emergency_stop() -> Result:
    """Halt the MCU immediately. Requires FIRMWARE_RESTART afterwards."""
    return _post("/printer/emergency_stop")


def firmware_restart() -> Result:
    return _post("/printer/firmware_restart")


# -- write: Z-offset (first-layer quality) --------------------------------

def babystep_z(delta: float) -> Result:
    """Nudge the live Z-offset by ``delta`` mm while moving (babystep).

    Positive = nozzle further from bed (fixes over-squished/split first layer).
    Negative = nozzle closer.
    """
    return gcode(f"SET_GCODE_OFFSET Z_ADJUST={delta:+.3f} MOVE=1")


def set_zoffset(value: float) -> Result:
    """Set the absolute live Z gcode offset."""
    return gcode(f"SET_GCODE_OFFSET Z={value:.3f} MOVE=1")


def save_zoffset() -> Result:
    """Bake the live offset into the saved probe z_offset and persist it.

    Folds the current gcode Z offset into the probe's z_offset, then SAVE_CONFIG
    (which restarts Klipper). Use after dialing in a good first layer so it sticks.
    """
    apply_res = gcode("Z_OFFSET_APPLY_PROBE")
    if not apply_res.ok:
        return apply_res
    return gcode("SAVE_CONFIG")
