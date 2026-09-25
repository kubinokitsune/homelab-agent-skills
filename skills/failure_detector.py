"""Failure detector -- train a small classifier on labeled print frames.

Scaffolded ahead of the dataset milestone (see print_recorder.training_readiness).
Run it once there are enough labeled prints:

    python -m skills.failure_detector        # train + report HONEST accuracy

It loads frames from the print dataset (label inherited per print: good vs bad),
extracts cheap image features (downscaled grayscale + edge maps + brightness
histogram -- no GPU, no deep nets), trains a RandomForest, and evaluates with
*leave-prints-out* grouping so correlated frames from one print can't inflate the
score. Saves the model for fast per-frame inference (`predict`) that replaces
moondream's slow, unreliable guessing.

Deps: Pillow + scikit-learn (both CPU-light).

Honest caveats baked in:
  * Frames inherit the print's single label -- early frames of a "bad" print
    look fine, so the bad class is noisy. Good enough for a first detector;
    per-frame labeling is a later refinement.
  * Needs a real spread (both classes, several prints) or it refuses to train.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from skills.config import config
from skills.logging import get_logger
from skills.result import Result
from skills import print_recorder

_log = get_logger("failure_detector")


def _frame_label(session_dir, frame_name: str, session_label):
    """The label to train on for one frame: its own swipe label if it has one,
    else the whole print's label. Per-frame beats per-print, since a print can
    be fine early and fail late."""
    per = print_recorder.get_frame_labels(session_dir).get(frame_name)
    return per if per in ("good", "bad") else session_label

FEATURE_SIZE = 32  # downscale frames to 32x32 grayscale


def _features(image_bytes: bytes):
    """Cheap feature vector for one frame, or None if it can't be decoded."""
    from PIL import Image, ImageFilter  # imported lazily so the module loads w/o Pillow
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("L").resize((FEATURE_SIZE, FEATURE_SIZE))
    except Exception:
        return None
    arr = np.asarray(im, dtype=np.float32) / 255.0
    edges = np.asarray(im.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    hist = np.histogram(arr, bins=16, range=(0, 1))[0] / arr.size
    summary = np.array([
        arr.mean(), arr.std(), edges.mean(), edges.std(),
        np.abs(np.diff(arr, axis=0)).mean(),   # vertical gradient (texture)
        np.abs(np.diff(arr, axis=1)).mean(),   # horizontal gradient
    ], dtype=np.float32)
    return np.concatenate([arr.flatten(), edges.flatten(), hist, summary])


def _load_dataset():
    """Return (X, y, groups). y: 1=bad/fail, 0=good. groups: one id per print."""
    X, y, groups = [], [], []
    root = config.print_dataset_dir
    if not root.is_dir():
        return np.array([]), np.array([]), np.array([])
    for i, s in enumerate(sorted(p for p in root.iterdir() if p.is_dir())):
        try:
            label = json.loads((s / "meta.json").read_text()).get("label")
        except Exception:
            label = None
        for f in sorted(s.glob("*.jpg")):
            # Per-frame swipe label wins; otherwise inherit the print's label.
            lab = _frame_label(s, f.name, label)
            # good/clean + imperfections (stringing/uneven) are all "not a failure"; bad = failure.
            if lab not in ("good", "stringing", "uneven", "bad"):
                continue
            feat = _features(f.read_bytes())
            if feat is not None:
                X.append(feat)
                y.append(1 if lab == "bad" else 0)
                groups.append(i)
    return np.array(X), np.array(y), np.array(groups)


# --- anomaly model: learn "clean" from good prints, no failures needed -------

ANOMALY_MIN_FRAMES = 80   # frames from clean prints before a baseline can train
# IsolationForest decision_function: higher = more normal. Banded into quality.
# FAILURE must sit BELOW the clean distribution's worst (~-0.082 observed), or tall
# prints under-represented in clean data read as failures. -0.12 gives margin; real
# spaghetti scores far more negative. Widen this back once clean data covers tall prints.
_CLEAN_ABOVE = 0.0
_FAILURE_BELOW = -0.12


def _score_to_pct(score: float) -> int:
    """Map the raw anomaly score to an intuitive 0-100 'print health %': anchored so
    the failure line reads ~30% and the clean line ~70% (perfect clean ~100%)."""
    pct = 30.0 + (score - _FAILURE_BELOW) * (70.0 - 30.0) / (_CLEAN_ABOVE - _FAILURE_BELOW)
    return int(max(0, min(100, round(pct))))


def _clean_frames():
    """Feature matrix from CLEAN/good-labeled prints -- the 'normal' class."""
    X = []
    root = config.print_dataset_dir
    if not root.is_dir():
        return np.array([])
    for s in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            label = json.loads((s / "meta.json").read_text()).get("label")
        except Exception:
            label = None
        for f in sorted(s.glob("*.jpg")):
            # A frame counts as "clean" if its own swipe label is good, or (no
            # swipe label) the whole print was labeled good.
            if _frame_label(s, f.name, label) != "good":
                continue
            feat = _features(f.read_bytes())
            if feat is not None:
                X.append(feat)
    return np.array(X)


def train_anomaly() -> Result:
    """Train an Isolation Forest on clean prints (needs NO failures)."""
    X = _clean_frames()
    if len(X) < ANOMALY_MIN_FRAMES:
        return Result.failure(
            f"need more clean prints: {len(X)}/{ANOMALY_MIN_FRAMES} good frames so far. "
            "Keep printing + `!label clean`.")
    from sklearn.ensemble import IsolationForest
    import joblib
    clf = IsolationForest(n_estimators=200, contamination=0.06, random_state=0).fit(X)
    config.print_anomaly_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_size": FEATURE_SIZE}, config.print_anomaly_model_path)
    return Result.success({"trained_on": len(X)})


def anomaly_readiness() -> Result:
    """Clean-frame count vs the minimum to (re)train the anomaly model -- cheap (no
    feature extraction), just counts .jpg in good-labeled prints."""
    root = config.print_dataset_dir
    n = 0
    if root.is_dir():
        for s in root.iterdir():
            if not s.is_dir():
                continue
            try:
                if json.loads((s / "meta.json").read_text()).get("label") == "good":
                    n += len(list(s.glob("*.jpg")))
            except Exception:
                continue
    return Result.success({
        "clean_frames": n, "need": ANOMALY_MIN_FRAMES,
        "ready": n >= ANOMALY_MIN_FRAMES,
        "trained": config.print_anomaly_model_path.exists(),
    })


def assess(image_bytes: bytes) -> Result:
    """Graded read of a frame -> {bucket: clean|imperfect|failure, score}.

    Uses the clean-print anomaly model: the further a frame is from 'normal
    clean print', the worse the bucket. Needs no failure examples to work.
    """
    p = config.print_anomaly_model_path
    if not p.exists():
        return Result.failure("no anomaly model yet -- run train_anomaly()")
    import joblib
    feat = _features(image_bytes)
    if feat is None:
        return Result.failure("couldn't decode frame")
    clf = joblib.load(p)["model"]
    s = float(clf.decision_function(feat.reshape(1, -1))[0])
    bucket = "clean" if s >= _CLEAN_ABOVE else ("imperfect" if s >= _FAILURE_BELOW else "failure")
    return Result.success({"bucket": bucket, "score": round(s, 3), "pct": _score_to_pct(s)})


def train(model_path: Path | None = None) -> Result:
    """Train + save the detector. Refuses (with a clear reason) on thin data."""
    model_path = Path(model_path or config.failure_model_path)
    X, y, groups = _load_dataset()
    n_prints = len(set(groups.tolist())) if len(groups) else 0
    n_good = int((y == 0).sum()) if len(y) else 0
    n_bad = int((y == 1).sum()) if len(y) else 0
    if n_prints < 4 or n_good == 0 or n_bad == 0:
        return Result.failure(
            f"not enough data: {n_prints} labeled prints "
            f"({n_good} good frames, {n_bad} bad frames). "
            "Need both classes and several prints — keep printing + labeling.")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    import joblib

    clf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=0)

    # Honest accuracy: hold out WHOLE prints each fold (frames are correlated).
    n_splits = min(5, n_prints)
    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    cv_acc = float(np.mean(accs))

    clf.fit(X, y)  # final fit on everything
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_size": FEATURE_SIZE}, model_path)
    return Result.success({
        "prints": n_prints, "good_frames": n_good, "bad_frames": n_bad,
        "cv_accuracy": round(cv_acc, 3), "model_path": str(model_path)})


def predict(image_bytes: bytes, model_path: Path | None = None) -> Result:
    """Classify a frame -> {failed: bool, confidence: float}. Needs a trained model."""
    model_path = Path(model_path or config.failure_model_path)
    if not model_path.exists():
        return Result.failure("no trained model yet -- run failure_detector.train()")
    import joblib
    feat = _features(image_bytes)
    if feat is None:
        return Result.failure("couldn't decode frame")
    clf = joblib.load(model_path)["model"]
    classes = list(clf.classes_)
    proba = clf.predict_proba(feat.reshape(1, -1))[0]
    p_fail = float(proba[classes.index(1)]) if 1 in classes else 0.0
    return Result.success({"failed": p_fail >= 0.5, "confidence": round(p_fail, 3)})


if __name__ == "__main__":
    res = train()
    print(res.data if res.ok else f"NOT TRAINED — {res.error}")
