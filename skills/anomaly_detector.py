"""Server anomaly detection -- Hermes's sixth sense.

UNSUPERVISED: no labels needed. Hermes records a metric snapshot every watchdog
sweep; once there's enough baseline, an Isolation Forest learns the *normal*
joint behavior of the box and flags readings that don't fit -- a memory leak
creeping RAM up, temps climbing, an abnormal load spike -- often before they
become an outright failure that the reactive watchdog would catch.

    anomaly_detector.record({"load1":0.1,"ram_pct":18,"disk_pct":7,"cpu_temp":59})
    anomaly_detector.train()                  # daily; needs >= MIN_TRAIN points
    res = anomaly_detector.score(metrics)     # {anomaly, score}

Metrics live in a rolling JSONL (config.metrics_path); the model in
config.anomaly_model_path. Deps: scikit-learn (already installed for Mason).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from skills.config import config
from skills.logging import get_logger
from skills.result import Result

_log = get_logger("anomaly_detector")

FEATURES = ["load1", "ram_pct", "disk_pct", "cpu_temp"]
MIN_TRAIN = 200      # snapshots before a baseline can be trained
MAX_KEEP = 8000      # rolling window (~11 days at one/2min) -- recent "normal"


def record(metrics: dict) -> Result:
    """Append one metric snapshot to the rolling history."""
    row = {"t": datetime.now().isoformat(timespec="seconds")}
    for k in FEATURES:
        row[k] = metrics.get(k)
    try:
        p = config.metrics_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return Result.success(True)
    except Exception as exc:
        return Result.failure(f"record failed: {exc}")


def _load_rows(limit: int = MAX_KEEP) -> list[dict]:
    p = config.metrics_path
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def _matrix(rows: list[dict]):
    import numpy as np
    X = [[r.get(k) for k in FEATURES] for r in rows
         if all(r.get(k) is not None for k in FEATURES)]
    return np.array(X, dtype=float) if X else None


def truncate() -> Result:
    """Trim the history file to the rolling window (keeps it bounded)."""
    rows = _load_rows(MAX_KEEP)
    try:
        with open(config.metrics_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return Result.success(len(rows))
    except Exception as exc:
        return Result.failure(f"truncate failed: {exc}")


def train() -> Result:
    """Train + save the Isolation Forest on the recorded normal baseline."""
    X = _matrix(_load_rows())
    n = 0 if X is None else len(X)
    if n < MIN_TRAIN:
        return Result.failure(f"not enough baseline yet: {n}/{MIN_TRAIN} snapshots "
                              "(Hermes records one every 2 min — give it time).")
    from sklearn.ensemble import IsolationForest
    import joblib
    clf = IsolationForest(n_estimators=150, contamination=0.02, random_state=0).fit(X)
    config.anomaly_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": FEATURES}, config.anomaly_model_path)
    return Result.success({"trained_on": n})


def score(metrics: dict) -> Result:
    """Score a reading against the baseline -> {anomaly: bool, score: float}.

    score < 0 = anomalous (the more negative, the weirder). Needs a trained model.
    """
    if not config.anomaly_model_path.exists():
        return Result.failure("no baseline model yet -- run train()")
    import numpy as np
    import joblib
    bundle = joblib.load(config.anomaly_model_path)
    x = np.array([[metrics.get(k) for k in bundle["features"]]], dtype=float)
    if np.isnan(x).any():
        return Result.failure("incomplete reading")
    clf = bundle["model"]
    s = float(clf.decision_function(x)[0])
    return Result.success({"anomaly": clf.predict(x)[0] == -1, "score": round(s, 3)})


def trends(n: int = 12) -> Result:
    """Recent metric snapshots (for !trends): list of the last n rows."""
    return Result.success(_load_rows(n))
