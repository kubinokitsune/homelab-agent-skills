"""The nightly librarian pass -- Axiom's real organizing work.

Three stages, ordered by destructiveness and each independently safe:

1. **Merge near-duplicates** (cosine >= ``MERGE_THRESHOLD`` *and* an LLM gate
   confirms they're genuinely redundant). The merge is **lossless**: the
   secondary note's full body is appended under a "Merged from" heading, then the
   secondary file is **moved to ``_Archive/``** (never deleted) with a tombstone
   pointing at the survivor. Its stale vector is dropped and the survivor
   re-embedded. Capped per run.

2. **Connect related-but-distinct** (``LINK_THRESHOLD`` <= cosine <
   ``MERGE_THRESHOLD``). These shouldn't be merged -- they cover related but
   different ground (the classic "homelab software" vs "homelab hardware"). So we
   *link* them: a managed ``Related`` block wrapped in ``%% %%`` markers (invisible
   in reading view, idempotent across runs). The note's own content is untouched.

3. **Subcategory MOCs** -- for each main node (top-level folder), cluster its
   notes into subcategories with the LLM and write/refresh ``00_MOC/<Folder>
   MOC.md``. Purely additive index notes.

``organize`` returns counts for the morning digest. ``dry_run=True`` reports what
*would* happen without writing anything -- always preview before trusting a run.

Raw conversation logs (``01_Conversations/``) are excluded from stages 1-2: they
are append-only records, not knowledge notes to edit.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

from skills import llm
from skills import obsidian_vault as vault
from skills import vault_index
from skills import vector_store as vs
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)

# -- tunables --------------------------------------------------------------
MERGE_THRESHOLD = 0.93   # at/above this, candidates for merging (after LLM gate)
LINK_THRESHOLD = 0.82    # [LINK, MERGE) band -> link, don't merge
MAX_MERGES = 5           # blast-radius cap per run
MAX_LINKS = 40           # notes connected per run

ARCHIVE_FOLDER = "_Archive"
MOC_FOLDER = "00_MOC"

# Folders never touched by the edit stages (merge + connect). Append-only logs,
# generated MOCs, the archive, and -- by Pipe's rule -- IB coursework, which must
# stay separate from personal projects (no merging IA notes, no cross-linking
# coursework into project notes).
NO_EDIT_PREFIXES = (ARCHIVE_FOLDER + "/", MOC_FOLDER + "/", "01_Conversations/", "School/")

# Folders excluded only from MOC building (generated/archive). Coursework is NOT
# here: School still gets its own internal MOC -- that organizes school notes
# among themselves, which doesn't mix them with anything.
_MOC_SKIP_PREFIXES = (ARCHIVE_FOLDER + "/", MOC_FOLDER + "/")

# Default "main nodes" to build subcategory MOCs for. Override via organize().
DEFAULT_MOC_FOLDERS = ["02_Topics", "School", "04_Projects", "05_Reference"]

# Managed "Related" block markers -- Obsidian comments, invisible in reading view.
_REL_START = "%% axiom-related-start %%"
_REL_END = "%% axiom-related-end %%"
_REL_BLOCK_RE = re.compile(re.escape(_REL_START) + r".*?" + re.escape(_REL_END), re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")


def _title(rel_path: str) -> str:
    """Wikilink target for a note -- Obsidian resolves these by basename."""
    return Path(rel_path).stem


def _protected(rel_path: str) -> bool:
    """True if a note is off-limits to the merge/connect (edit) stages."""
    return rel_path.startswith(NO_EDIT_PREFIXES)


# -- stage 1: merge near-duplicates ----------------------------------------

def _confirm_merge(note_a: dict, note_b: dict, model: str | None) -> bool:
    """LLM safety gate: only merge if the model judges the two notes genuinely
    redundant (same subject, one supersedes the other), not merely related."""
    a_title, b_title = _title(note_a["path"]), _title(note_b["path"])
    prompt = (
        "Two Obsidian notes look similar. Decide if they are REDUNDANT -- covering "
        "the same subject such that they should be merged into one note -- or merely "
        "RELATED (same area but different focus, which should stay separate and just "
        "be linked).\n\n"
        f"NOTE A: {a_title}\n{note_a['body'][:1200]}\n\n"
        f"NOTE B: {b_title}\n{note_b['body'][:1200]}\n\n"
        "Answer with a single word: MERGE (truly redundant) or KEEP (related but "
        "distinct). When unsure, answer KEEP."
    )
    res = llm.chat(prompt, system="You are a careful librarian. Default to KEEP.", model=model)
    if not res.ok:
        log.warning("merge gate LLM failed, defaulting to KEEP: %s", res.error)
        return False
    return res.data.strip().upper().startswith("MERGE")


def _do_merge(collection: str, primary: str, secondary: str,
              np: dict, ns: dict) -> None:
    """Fold ``secondary`` into ``primary`` losslessly, then archive secondary."""
    block = (
        f"\n\n---\n\n## Merged from [[{_title(secondary)}]]\n"
        f"_Folded in by Axiom on {date.today().isoformat()}; original archived at "
        f"`{ARCHIVE_FOLDER}/{secondary}`._\n\n"
        f"{ns['body'].strip()}\n"
    )
    vault.write_note(primary, np["body"].rstrip() + block, np["frontmatter"], overwrite=True)

    # Archive the secondary (move, don't delete) and prepend a tombstone.
    arch_path = f"{ARCHIVE_FOLDER}/{secondary}"
    mv = vault.move_note(secondary, arch_path, overwrite=True)
    if not mv.ok:
        log.warning("couldn't archive %s: %s", secondary, mv.error)
        return
    moved = vault.read_note(arch_path)
    if moved.ok:
        tomb = (
            f"> [!info] Archived {date.today().isoformat()} -- merged into "
            f"[[{_title(primary)}]]. Kept here for recovery.\n\n{moved.data['body']}"
        )
        vault.write_note(arch_path, tomb, moved.data["frontmatter"], overwrite=True)

    # Index hygiene: drop the archived vector, refresh the survivor.
    vs.delete(collection, secondary)
    vault_index.index_one(collection, primary)


def merge_duplicates(collection: str, dry_run: bool, model: str | None) -> list[dict]:
    """Return the merges performed (or that would be): [{kept, archived, score}]."""
    pairs = (vs.find_duplicate_pairs(collection, MERGE_THRESHOLD, max_pairs=25).data) or []
    done: list[dict] = []
    touched: set[str] = set()
    for pair in pairs:
        if len(done) >= MAX_MERGES:
            break
        a, b, score = pair.get("a"), pair.get("b"), pair.get("score")
        if not a or not b or _protected(a) or _protected(b):
            continue
        if a in touched or b in touched:  # don't chain-merge in one pass
            continue
        na, nb = vault.read_note(a), vault.read_note(b)
        if not (na.ok and nb.ok):
            continue
        if not _confirm_merge(na.data, nb.data, model):
            continue
        # Keep the longer note as primary; archive the shorter.
        if len(nb.data["body"]) > len(na.data["body"]):
            a, b, na, nb = b, a, nb, na
        if not dry_run:
            _do_merge(collection, a, b, na.data, nb.data)
        done.append({"kept": a, "archived": b, "score": score})
        touched.update((a, b))
        log.info("%smerged %s <- %s (%.3f)", "[dry] " if dry_run else "", a, b, score or 0)
    return done


# -- stage 2: connect related notes ----------------------------------------

def _existing_related(body: str) -> tuple[str, set[str]]:
    """Return (body_without_block, set_of_linked_titles) for the managed block."""
    m = _REL_BLOCK_RE.search(body)
    if not m:
        return body, set()
    linked = set(_WIKILINK_RE.findall(m.group(0)))
    cleaned = (body[: m.start()] + body[m.end():]).rstrip()
    return cleaned, linked


def _write_related_block(note_path: str, target_titles: set[str]) -> bool:
    """Add target_titles to the note's managed Related block. Idempotent.
    Returns True if the note gained at least one new link."""
    n = vault.read_note(note_path)
    if not n.ok:
        return False
    base, existing = _existing_related(n.data["body"])
    # Don't link a note to itself, and only count genuinely new links.
    targets = {t for t in target_titles if t and t != _title(note_path)}
    merged = existing | targets
    if merged == existing:
        return False
    items = "\n".join(f"- [[{t}]]" for t in sorted(merged))
    block = f"{_REL_START}\n## 🔗 Related\n{items}\n{_REL_END}"
    new_body = f"{base.rstrip()}\n\n{block}\n"
    vault.write_note(note_path, new_body, n.data["frontmatter"], overwrite=True)
    return True


def connect_related(collection: str, dry_run: bool) -> int:
    """Link related-but-distinct notes. Returns the count of notes connected."""
    pairs = (vs.find_duplicate_pairs(collection, LINK_THRESHOLD, max_pairs=300).data) or []
    adjacency: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        score = p.get("score") or 0
        if score >= MERGE_THRESHOLD:  # that band is for merging, not linking
            continue
        a, b = p.get("a"), p.get("b")
        if not a or not b or _protected(a) or _protected(b):
            continue
        adjacency[a].add(_title(b))
        adjacency[b].add(_title(a))

    connected = 0
    for note_path, titles in adjacency.items():
        if connected >= MAX_LINKS:
            break
        if dry_run:
            connected += 1
            continue
        if _write_related_block(note_path, titles):
            connected += 1
    return connected


# -- stage 3: subcategory MOCs ---------------------------------------------

def _parse_json_object(text: str) -> dict:
    """Best-effort extraction of the first JSON object from an LLM reply."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else {}
    except ValueError:
        return {}


def _cluster_titles(folder: str, note_paths: list[str], model: str | None) -> dict[str, list[str]]:
    """Ask the LLM to group note titles into subcategories. Every title is placed
    exactly once; anything the model drops lands in 'Other'."""
    titles = [_title(p) for p in note_paths]
    prompt = (
        f"These note titles live in the '{folder}' section of an Obsidian vault. "
        "Group them into 3-7 meaningful subcategories. Return ONLY a JSON object "
        "mapping each subcategory name to an array of the EXACT titles given. Use "
        "each title exactly once.\n\nTitles:\n" + "\n".join(f"- {t}" for t in titles)
    )
    res = llm.chat(prompt, system="You organize notes. Output only valid JSON.", model=model)
    groups = _parse_json_object(res.data) if res.ok else {}
    known = set(titles)
    placed: set[str] = set()
    cleaned: dict[str, list[str]] = {}
    for cat, arr in groups.items():
        if not isinstance(arr, list):
            continue
        keep = [t for t in arr if t in known and t not in placed]
        for t in keep:
            placed.add(t)
        if keep:
            cleaned[str(cat)] = keep
    leftovers = [t for t in titles if t not in placed]
    if leftovers:
        cleaned.setdefault("Other", []).extend(leftovers)
    return cleaned or {"All Notes": titles}


def _render_moc(folder: str, groups: dict[str, list[str]]) -> str:
    lines = [
        f"# {folder} -- Map of Content",
        "",
        f"> Auto-maintained by Axiom. Last refresh: {date.today().isoformat()}.",
        "",
    ]
    for cat, titles in groups.items():
        lines.append(f"## {cat}")
        lines.extend(f"- [[{t}]]" for t in sorted(titles))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_mocs(main_folders: list[str], dry_run: bool, model: str | None) -> tuple[int, int]:
    """Write/refresh a subcategory MOC per main node. Returns (mocs, subcategories)."""
    mocs = 0
    subcats = 0
    for folder in main_folders:
        listed = vault.list_notes(folder=folder)
        if not listed.ok:
            continue
        notes = [n for n in listed.data if not n.startswith(_MOC_SKIP_PREFIXES)]
        if len(notes) < 4:  # too few to be worth a MOC
            continue
        groups = _cluster_titles(folder, notes, model)
        subcats += len(groups)
        mocs += 1
        if not dry_run:
            vault.write_note(
                f"{MOC_FOLDER}/{folder} MOC",
                _render_moc(folder, groups),
                {"title": f"{folder} MOC", "type": "moc", "tags": ["moc", "index", "auto"]},
                overwrite=True,
            )
        log.info("%sMOC for %s: %d subcategories", "[dry] " if dry_run else "", folder, len(groups))
    return mocs, subcats


# Note types stamped by agents (vs. Pipe's own notes) -- the ones to catalogue.
AGENT_NOTE_TYPES = {
    "source", "source-synthesis", "forge-note", "shopping-list", "build-log",
    "build-tracker", "forge-review", "forge-daily",
}


def build_agent_moc(dry_run: bool) -> int:
    """Catalogue agent-created notes by kind into one connected MOC (additive).

    Every agent note now carries a consistent ``type`` + stable ``uid``, so this
    groups them into named categories in a single index that links them together
    -- one place for Pipe/Codex/Axiom to find anything an agent made.
    """
    cats: dict[str, list] = defaultdict(list)
    listed = vault.list_notes()
    if not listed.ok:
        return 0
    for rel in listed.data:
        if rel.startswith(_MOC_SKIP_PREFIXES):
            continue
        n = vault.read_note(rel)
        if not n.ok:
            continue
        t = n.data["frontmatter"].get("type")
        # Catalogue known agent note types AND any "<agent>-note" (every agent can
        # now write notes via the shared base, e.g. scout-note, chiron-note).
        if t in AGENT_NOTE_TYPES or (t and t.endswith("-note")):
            cats[t].append((_title(rel), n.data["frontmatter"].get("uid")))
    total = sum(len(v) for v in cats.values())
    if total == 0:
        return 0
    lines = ["# Agent Notes MOC", "",
             "_Notes your agents created, grouped by kind and cross-indexed. "
             "Auto-maintained by Axiom; each entry shows its stable `uid`._", ""]
    for t in sorted(cats):
        lines.append(f"## {t}  ({len(cats[t])})")
        for title, uid in sorted(cats[t]):
            lines.append(f"- [[{title}]]" + (f"  `{uid}`" if uid else ""))
        lines.append("")
    if not dry_run:
        vault.write_note(f"{MOC_FOLDER}/Agent Notes MOC", "\n".join(lines),
                         {"title": "Agent Notes MOC", "type": "moc",
                          "tags": ["moc", "agents", "auto"]}, overwrite=True)
    log.info("%sagent-notes MOC: %d notes across %d categories",
             "[dry] " if dry_run else "", total, len(cats))
    return total


# -- orchestrator ----------------------------------------------------------

@skill
def organize(
    collection: str,
    main_folders: list[str] | None = None,
    dry_run: bool = False,
    model: str | None = None,
) -> Result:
    """Run the full nightly librarian pass. Each stage is isolated so one failure
    doesn't sink the rest. Returns counts for the morning digest."""
    folders = main_folders if main_folders is not None else DEFAULT_MOC_FOLDERS
    summary = {"merged": [], "connected": 0, "mocs": 0, "subcats": 0,
               "agent_notes": 0, "dry_run": dry_run}

    # Stage 0: make sure every note has a stable uid before anything moves it.
    if not dry_run:
        try:
            vault.ensure_uids()
        except Exception:
            log.exception("uid backfill failed")
    try:
        summary["merged"] = merge_duplicates(collection, dry_run, model)
    except Exception:
        log.exception("merge stage failed")
    try:
        summary["connected"] = connect_related(collection, dry_run)
    except Exception:
        log.exception("connect stage failed")
    try:
        summary["mocs"], summary["subcats"] = build_mocs(folders, dry_run, model)
    except Exception:
        log.exception("MOC stage failed")
    try:
        summary["agent_notes"] = build_agent_moc(dry_run)
    except Exception:
        log.exception("agent-notes MOC stage failed")

    log.info(
        "organize done%s: merged=%d connected=%d mocs=%d subcats=%d",
        " [dry]" if dry_run else "", len(summary["merged"]),
        summary["connected"], summary["mocs"], summary["subcats"],
    )
    return Result.success(summary)
