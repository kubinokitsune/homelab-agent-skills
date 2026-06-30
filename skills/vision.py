"""Vision -- Mason's eyes on the print.

Grabs a single JPEG frame from the printer camera and asks a *local* vision
model (moondream, via Ollama) about it. No cloud, no GPU, no separate ML
service -- it reuses the Ollama that already runs the agents' brains.

    res = vision.assess_print()
    if res.ok:
        print(res.data["verdict"])     # "OK ..." or "FAIL ..."
        img = res.data["frame"]        # raw JPEG bytes (post to Discord)

Synchronous like the rest of the library; call from an async agent via
``asyncio.to_thread(vision.assess_print)``. Vision inference on a CPU is slow
(tens of seconds with moondream), so this is meant for periodic checks and
on-demand looks -- not every frame.
"""

from __future__ import annotations

import base64
import json
import urllib.request

from skills.config import config
from skills.logging import get_logger
from skills.result import Result

_log = get_logger("vision")


def grab_frame(timeout: float = 10.0) -> Result:
    """Fetch a single JPEG frame from the camera. Returns raw bytes on success."""
    try:
        with urllib.request.urlopen(config.camera_snapshot_url, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            return Result.failure("camera returned an empty frame")
        return Result.success(data)
    except Exception as exc:
        return Result.failure(f"camera grab failed: {exc}")


def analyze(question: str, image: bytes, model: str | None = None,
            timeout: float = 180.0) -> Result:
    """Ask the local vision model a question about an image. Returns its text.

    CPU inference is slow, hence the generous timeout.
    """
    model = model or config.vision_model
    payload = {
        "model": model,
        "prompt": question,
        "images": [base64.b64encode(image).decode("ascii")],
        "stream": False,
        "keep_alive": "10m",
        # Match threads to the cpuset (see config.ollama_num_threads) -- also
        # speeds up moondream and stops it pinning the box during a look.
        "options": {"num_thread": config.ollama_num_threads},
    }
    try:
        req = urllib.request.Request(
            config.ollama_host.rstrip("/") + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        return Result.success((d.get("response") or "").strip())
    except Exception as exc:
        return Result.failure(f"vision analysis failed: {exc}")


# Structured prompt: forces a leading OK/FAIL token the watchdog can branch on.
_PRINT_HEALTH_Q = (
    "This is a webcam photo looking down at a 3D printer's bed during a print. "
    "Look carefully. Is the print still cleanly attached to the bed, or has it "
    "detached and turned into a messy tangle of stray plastic strands (a "
    "'spaghetti' failure)?\n"
    "Answer in this exact form: first word FAIL if you see a detached print or a "
    "tangle of stray filament strands, otherwise OK. Then one short sentence "
    "describing what you see."
)


def assess_print(model: str | None = None) -> Result:
    """Grab a frame and get a print-health read. data = {frame, verdict, failed}."""
    fr = grab_frame()
    if not fr.ok:
        return fr
    res = analyze(_PRINT_HEALTH_Q, fr.data, model)
    if not res.ok:
        return res
    verdict = res.data
    failed = verdict.strip().upper().startswith("FAIL")
    return Result.success({"frame": fr.data, "verdict": verdict, "failed": failed})
