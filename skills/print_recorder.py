"""Print dataset recorder -- builds the training set for a failure detector.

During a print, Mason saves a frame every ~20s into a per-print folder. When
the print ends, Pipe labels it good/bad. Over normal printing this accumulates
a labeled dataset of *this* printer, *this* camera, *this* lighting -- which is
what a small, reliable, CPU-friendly failure classifier trains on later (the way
Obico/Spaghetti Detective was built).

Cheap by design: saving a JPEG costs nothing (no vision/LLM/CPU load) -- only
``vision.assess_print`` is heavy, and the recorder never calls it.

Layout::

    print-dataset/
      20260625_224501_3dbenchy.gcode/
        meta.json            # filename, outcome, label, frame count
        frame_000000.jpg     # named by seconds-into-print
        frame_000020.jpg
        ...
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from skills.config import config
from skills.logging import get_logger
from skills.result import Result

_log = get_logger("print_recorder")

# Richer quality labels (aliases -> canonical). `good` = clean; `stringing` and
# `uneven` are imperfections (still not failures); `bad` = a real failure.
_LABEL_ALIASES = {
    "good": "good", "clean": "good", "success": "good", "ok": "good",
    "stringing": "stringing", "hairs": "stringing", "hair": "stringing",
    "strings": "stringing", "clumps": "stringing", "blobs": "stringing",
    "uneven": "uneven", "layers": "uneven", "layer": "uneven", "wavy": "uneven",
    "bad": "bad", "spaghetti": "bad", "fail": "bad", "failed": "bad", "detached": "bad",
}


def normalize_label(s: str) -> str | None:
    """Map a user label (good/clean/stringing/uneven/bad/...) to its canonical form."""
    return _LABEL_ALIASES.get((s or "").strip().lower())


def _root() -> Path:
    d = config.print_dataset_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in (name or "print"))[:40]


def start_session(filename: str) -> Result:
    """Begin a recording session for a print. Returns the session folder path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = _root() / f"{ts}_{_safe(filename)}"
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"started": ts, "filename": filename, "frames": 0,
            "outcome": None, "label": None}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return Result.success(str(folder))


def save_frame(session: str, image: bytes, elapsed: float) -> Result:
    """Save one frame into the session, named by seconds-into-print."""
    try:
        folder = Path(session)
        (folder / f"frame_{int(elapsed):06d}.jpg").write_bytes(image)
        meta_p = folder / "meta.json"
        meta = json.loads(meta_p.read_text())
        meta["frames"] = meta.get("frames", 0) + 1
        meta_p.write_text(json.dumps(meta, indent=2))
        return Result.success(meta["frames"])
    except Exception as exc:
        return Result.failure(f"save_frame failed: {exc}")


def end_session(session: str, outcome: str) -> Result:
    """Close a session, recording how the print ended (complete/cancelled/error)."""
    try:
        folder = Path(session)
        meta_p = folder / "meta.json"
        meta = json.loads(meta_p.read_text())
        meta["outcome"] = outcome
        meta["ended"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_p.write_text(json.dumps(meta, indent=2))
        return Result.success(meta)
    except Exception as exc:
        return Result.failure(f"end_session failed: {exc}")


def label_last(label: str) -> Result:
    """Tag the most recent print: clean(good) / stringing / uneven / bad."""
    canon = normalize_label(label)
    if not canon:
        return Result.failure("label not recognized — use clean, stringing, uneven, or bad")
    sessions = sorted((p for p in _root().iterdir() if p.is_dir()),
                      key=lambda p: p.stat().st_mtime)
    if not sessions:
        return Result.failure("no recorded prints yet")
    meta_p = sessions[-1] / "meta.json"
    meta = json.loads(meta_p.read_text())
    meta["label"] = canon
    meta_p.write_text(json.dumps(meta, indent=2))
    return Result.success({"folder": sessions[-1].name, "frames": meta.get("frames", 0),
                           "label": canon})


def today_summary() -> Result:
    """Today's recorded prints: count, frames, and each print's outcome/label."""
    today = datetime.now().strftime("%Y%m%d")
    sessions = sorted(p for p in _root().iterdir()
                      if p.is_dir() and p.name.startswith(today))
    prints, frames = [], 0
    for s in sessions:
        try:
            m = json.loads((s / "meta.json").read_text())
        except Exception:
            m = {}
        fr = len(list(s.glob("*.jpg")))
        frames += fr
        prints.append({"filename": m.get("filename", s.name),
                       "outcome": m.get("outcome"), "label": m.get("label"),
                       "frames": fr})
    return Result.success({"count": len(sessions), "frames": frames, "prints": prints})


def training_readiness() -> Result:
    """Is the dataset big enough to train a failure detector yet?

    data = {ready, good, bad, need_good, need_bad}. 'ready' once both thresholds
    (config.train_min_good / train_min_bad) are met.
    """
    s = stats().data
    need_good, need_bad = config.train_min_good, config.train_min_bad
    return Result.success({
        "ready": s["good"] >= need_good and s["bad"] >= need_bad,
        "good": s["good"], "bad": s["bad"],
        "need_good": need_good, "need_bad": need_bad,
    })


def _flag() -> "Path":
    return _root() / ".train_ready_announced"


def already_announced() -> bool:
    """True if we've already pinged that training is unlocked (so we don't nag)."""
    return _flag().exists()


def mark_announced() -> Result:
    """Record that the 'ready to train' milestone has been announced once."""
    _flag().write_text("1")
    return Result.success(True)


def stats() -> Result:
    """Dataset summary: prints, frames, disk used, and label breakdown."""
    root = _root()
    sessions = [p for p in root.iterdir() if p.is_dir()]
    frames = size = 0
    good = bad = unlabeled = 0
    for s in sessions:
        for f in s.glob("*.jpg"):
            frames += 1
            size += f.stat().st_size
        try:
            lab = json.loads((s / "meta.json").read_text()).get("label")
        except Exception:
            lab = None
        if lab == "good":
            good += 1
        elif lab == "bad":
            bad += 1
        else:
            unlabeled += 1
    return Result.success({
        "prints": len(sessions), "frames": frames,
        "size_mb": round(size / 1e6, 1),
        "good": good, "bad": bad, "unlabeled": unlabeled,
    })
