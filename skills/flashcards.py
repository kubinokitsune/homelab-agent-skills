"""Flashcards with spaced repetition (SM-2) -- Chiron's study system.

Cards live in one JSON store. Each review reschedules a card with the SM-2
algorithm (the one Anki is based on): get it right easily and it won't come back
for a while; miss it and it returns tomorrow. A study log tracks daily streaks.

    flashcards.add_card("Avogadro's number?", "6.022e23", "Chemistry")
    due = flashcards.due_cards("Chemistry").data
    flashcards.grade(card_id, "good")     # again | hard | good | easy
    flashcards.streak().data              # {streak, total_days}
"""

from __future__ import annotations

import json
import uuid
from datetime import date, timedelta

from skills.config import config
from skills.logging import get_logger
from skills.result import Result

_log = get_logger("flashcards")

# Rating -> SM-2 quality (0-5). <3 = a miss (resets the card to tomorrow).
_QUALITY = {"again": 0, "hard": 3, "good": 4, "easy": 5}


def _today() -> str:
    return date.today().isoformat()


def _load() -> dict:
    p = config.flashcards_path
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"cards": [], "study_log": []}


def _save(data: dict) -> None:
    p = config.flashcards_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def add_card(front: str, back: str, subject: str = "") -> Result:
    data = _load()
    card = {"id": uuid.uuid4().hex[:6], "front": front.strip(), "back": back.strip(),
            "subject": (subject.strip() or "General"), "ease": 2.5,
            "interval": 0, "reps": 0, "due": _today()}
    data["cards"].append(card)
    _save(data)
    return Result.success(card)


def get_card(card_id: str) -> Result:
    for c in _load()["cards"]:
        if c["id"] == card_id:
            return Result.success(c)
    return Result.failure("card not found")


def due_cards(subject: str | None = None) -> Result:
    today = _today()
    cards = [c for c in _load()["cards"] if c["due"] <= today]
    if subject:
        cards = [c for c in cards if c["subject"].lower() == subject.lower()]
    return Result.success(cards)


def grade(card_id: str, rating: str) -> Result:
    """Apply SM-2 to a reviewed card and reschedule it."""
    q = _QUALITY.get(rating)
    if q is None:
        return Result.failure("rating must be again/hard/good/easy")
    data = _load()
    card = next((c for c in data["cards"] if c["id"] == card_id), None)
    if not card:
        return Result.failure("card not found")
    if q < 3:  # missed -> see it again tomorrow, keep the streak of reps reset
        card["reps"], card["interval"] = 0, 1
    else:
        if card["reps"] == 0:
            card["interval"] = 1
        elif card["reps"] == 1:
            card["interval"] = 6
        else:
            card["interval"] = max(1, round(card["interval"] * card["ease"]))
        card["reps"] += 1
        card["ease"] = max(1.3, card["ease"] + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
    card["due"] = (date.today() + timedelta(days=card["interval"])).isoformat()
    _save(data)
    return Result.success({"interval": card["interval"], "due": card["due"]})


def subjects() -> Result:
    today = _today()
    out: dict[str, dict] = {}
    for c in _load()["cards"]:
        s = c["subject"]
        out.setdefault(s, {"total": 0, "due": 0})
        out[s]["total"] += 1
        if c["due"] <= today:
            out[s]["due"] += 1
    return Result.success(out)


def log_study(subject: str = "") -> Result:
    """Record that Pipe studied today (for the streak)."""
    data = _load()
    if _today() not in data["study_log"]:
        data["study_log"].append(_today())
        _save(data)
    return streak()


def streak() -> Result:
    """Consecutive days studied, ending today or yesterday."""
    days = set(_load().get("study_log", []))
    if not days:
        return Result.success({"streak": 0, "total_days": 0})
    d = date.today()
    if d.isoformat() not in days:      # let the streak survive until end of today
        d -= timedelta(days=1)
    s = 0
    while d.isoformat() in days:
        s += 1
        d -= timedelta(days=1)
    return Result.success({"streak": s, "total_days": len(days)})


def stats() -> Result:
    today = _today()
    cards = _load()["cards"]
    return Result.success({
        "total": len(cards),
        "due": sum(1 for c in cards if c["due"] <= today),
        "subjects": subjects().data,
        "streak": streak().data["streak"],
    })
