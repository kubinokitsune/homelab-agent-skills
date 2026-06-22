"""ML auto-tagging for the vault -- Axiom's nearest-centroid classifier + spaCy.

Two signals, existing-vocabulary-first so the vault's tag set stays controlled
instead of sprawling:

1. **Nearest-centroid classifier (embeddings).** Each existing tag with enough
   example notes gets a centroid = the mean embedding of the notes carrying it.
   An under-tagged note is assigned the tags whose centroid its own embedding
   sits closest to -- reusing Pipe's vocabulary and learning each tag's meaning
   from how it's actually used. This is the precision workhorse.

2. **spaCy keyphrase extraction (NER + noun chunks).** Surfaces the specific
   terms a note is about; used to reinforce vocabulary matches and to propose a
   small number of genuinely new tags the vocabulary is missing.

Tagging is **additive only** (never removes a tag Pipe set) and capped per note.
``dry_run`` previews without writing.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from skills import obsidian_vault as vault
from skills import vector_store as vs
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)

MIN_TAG_EXAMPLES = 3      # a tag needs this many notes to form a usable centroid
# Tags are selected note-relative: take the best-fitting tag (if it clears the
# floor), then only others within MARGIN of it. This adapts to each note's own
# score scale -- a sharp note keeps several tags, a diffuse one keeps just its top.
CENTROID_FLOOR = 0.60     # the best tag must reach at least this cosine to tag at all
CENTROID_MARGIN = 0.06    # include further tags only within this of the top score
MAX_ADD = 3              # cap tags added to any one note
MAX_NEW_PER_NOTE = 0      # out-of-vocabulary (spaCy-invented) tags; off by default
UNDERTAGGED_MAX = 1       # by default only tag notes with <= this many tags
SKIP_PREFIXES = ("_Archive/", "00_MOC/")

# spaCy entity labels worth turning into tags.
_ENT_LABELS = {"ORG", "PRODUCT", "PERSON", "GPE", "LOC", "EVENT",
               "WORK_OF_ART", "FAC", "NORP", "LANGUAGE"}
# Too-generic words to never emit as tags.
_GENERIC = {"thing", "things", "note", "notes", "stuff", "way", "time", "lot",
            "bit", "part", "number", "example", "idea", "today", "day", "week",
            "year", "people", "person", "place", "work", "use", "kind", "type"}

_nlp = None


def _get_nlp():
    """Lazy-load spaCy once per process (model load is ~1s)."""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _norm_tag(s: str) -> str:
    """Normalize text to the vault's tag form: lowercase, hyphenated, alnum."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _ok_tag(t: str) -> bool:
    return 3 <= len(t) <= 30 and not t.isdigit() and t not in _GENERIC


def _keyphrases(body: str, nlp) -> Counter:
    """Candidate tag -> weight from a note's entities and noun-chunk heads."""
    doc = nlp(body[:4000])
    cand: Counter = Counter()
    for ent in doc.ents:
        if ent.label_ in _ENT_LABELS:
            t = _norm_tag(ent.text)
            if _ok_tag(t):
                cand[t] += 2  # entities weighted higher than plain chunks
    for chunk in doc.noun_chunks:
        words = [w.lemma_.lower() for w in chunk
                 if w.is_alpha and not w.is_stop and len(w) > 2]
        if not words:
            continue
        t = _norm_tag("-".join(words[-2:]))  # keep the chunk head (last 2 words)
        if _ok_tag(t):
            cand[t] += 1
    return cand


def _build_centroids(vecs: dict, tag_notes: dict, min_examples: int) -> dict:
    import numpy as np
    cents = {}
    for tag, paths in tag_notes.items():
        mats = [vecs[p] for p in paths if p in vecs]
        if len(mats) >= min_examples:
            c = np.asarray(mats, dtype=float).mean(axis=0)
            n = np.linalg.norm(c)
            if n > 0:
                cents[tag] = c / n
    return cents


def _centroid_tags(note_vec, cents: dict, floor: float, margin: float, top: int) -> list[str]:
    """Top-plus-margin selection: the best tag (if >= floor), then tags within
    ``margin`` of it. Adapts to each note's own score scale."""
    import numpy as np
    v = np.asarray(note_vec, dtype=float)
    nv = np.linalg.norm(v)
    if nv == 0:
        return []
    v = v / nv
    scored = sorted(((tag, float(v @ c)) for tag, c in cents.items()), key=lambda x: -x[1])
    if not scored or scored[0][1] < floor:
        return []
    cut = scored[0][1] - margin
    return [t for t, s in scored if s >= cut and s >= floor][:top]


@skill
def autotag(dry_run: bool = True, only_undertagged: bool = True,
            max_add: int = MAX_ADD, max_new: int = MAX_NEW_PER_NOTE) -> Result:
    """Suggest/apply tags. Returns counts + per-note ``{path, added}`` suggestions.

    ``only_undertagged`` limits work to notes with <= UNDERTAGGED_MAX tags (the
    notes that actually need it). Additive: existing tags are always kept.
    ``max_new`` > 0 lets spaCy propose out-of-vocabulary tags (off by default).
    """
    notes = vault.list_notes().data or []

    # Build vocabulary: tag -> notes carrying it, and each note's current tags.
    tag_notes: dict[str, list[str]] = defaultdict(list)
    note_tags: dict[str, list[str]] = {}
    for rel in notes:
        n = vault.read_note(rel)
        if not n.ok:
            continue
        tags = (n.data["frontmatter"] or {}).get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        tags = [str(t).lower() for t in tags]
        note_tags[rel] = tags
        for t in tags:
            tag_notes[t].append(rel)

    vocab = set(tag_notes)
    vecs = vs.all_vectors("vault_index").data or {}
    cents = _build_centroids(vecs, tag_notes, MIN_TAG_EXAMPLES)

    results = []
    tags_added = 0
    for rel in notes:
        if rel.startswith(SKIP_PREFIXES):
            continue
        existing = note_tags.get(rel, [])
        if only_undertagged and len(existing) > UNDERTAGGED_MAX:
            continue
        n = vault.read_note(rel)
        if not n.ok:
            continue
        body = n.data["body"].strip()
        if not body:
            continue

        # Existing-vocabulary tags by centroid fit (the precision workhorse).
        ctags = _centroid_tags(vecs[rel], cents, CENTROID_FLOOR, CENTROID_MARGIN, max_add) if rel in vecs else []
        suggested: list[str] = [t for t in ctags if t not in existing]

        # Optional: let spaCy propose a couple of genuinely new tags (off by default).
        if max_new > 0:
            cand = _keyphrases(body, _get_nlp())
            added_new = 0
            for t, c in cand.most_common():
                if added_new >= max_new:
                    break
                if c >= 2 and t not in vocab and t not in existing and t not in suggested:
                    suggested.append(t)
                    added_new += 1

        suggested = suggested[:max_add]
        if not suggested:
            continue

        results.append({"path": rel, "added": suggested})
        tags_added += len(suggested)
        if not dry_run:
            fm = dict(n.data["frontmatter"] or {})
            fm["tags"] = existing + suggested
            vault.write_note(rel, n.data["body"], fm, overwrite=True)

    log.info("autotag%s: %d notes, %d tags", " [dry]" if dry_run else "", len(results), tags_added)
    return Result.success({
        "notes_tagged": len(results), "tags_added": tags_added,
        "dry_run": dry_run, "results": results,
    })
