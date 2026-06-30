"""Read and write the Obsidian vault -- the agents' shared brain.

Agents leave each other context here and Iris builds the digest from it. The
vault is a populated personal vault, so this skill is built to be safe in it:

* **Structure-agnostic.** Every function takes a path *relative to the vault
  root* (``"04_Projects/foo.md"``). The skill imposes no folder layout -- agents
  decide where they write.
* **Never clobbers.** ``write_note`` refuses to overwrite an existing note
  unless ``overwrite=True``. Logs/running lists use ``append_note`` instead.
* **Stays inside the vault.** Paths are resolved and checked against the vault
  root, so an agent (or an LLM feeding one) can't write to ``../../somewhere``.
* **Frontmatter-aware.** YAML frontmatter is parsed on read and written on
  create, matching the vault's existing ``type/tags/created/updated`` convention.

All paths use ``.md`` implicitly -- pass it or omit it, both work.
"""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import yaml


def new_uid() -> str:
    """A short, stable note identifier (8 hex chars) -- survives renames/moves."""
    return uuid.uuid4().hex[:8]

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)


# -- path safety -----------------------------------------------------------

def _resolve(relative_path: str, as_dir: bool = False) -> Path:
    """Resolve a vault-relative path to an absolute one, confined to the vault.

    For note paths (``as_dir=False``) a ``.md`` suffix is added if missing. For
    folder paths (``as_dir=True``) the name is left alone. Raises ValueError if
    the result would escape the vault root (path traversal), which @skill turns
    into a clean Result.failure.
    """
    vault = config.vault_path.resolve()
    candidate = (vault / relative_path).resolve()
    if not as_dir and candidate.suffix == "":
        candidate = candidate.with_suffix(".md")
    if vault not in candidate.parents and candidate != vault:
        raise ValueError(f"path '{relative_path}' escapes the vault root")
    return candidate


def _relativize(path: Path) -> str:
    """Vault-relative string form of an absolute path, for return payloads."""
    return path.relative_to(config.vault_path.resolve()).as_posix()


# -- frontmatter -----------------------------------------------------------

def _split_frontmatter(raw: str) -> tuple[dict, str]:
    """Split raw note text into (frontmatter dict, body). Empty dict if none."""
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                # Malformed frontmatter (e.g. an unquoted colon in a value) --
                # don't let it make the note invisible. Treat as no frontmatter
                # and keep the body so the note stays readable and searchable.
                meta = {}
            if isinstance(meta, dict):
                return meta, parts[2].lstrip("\n")
    return {}, raw


def _compose(frontmatter: dict | None, body: str) -> str:
    """Render frontmatter + body back into note text."""
    if not frontmatter:
        return body
    fm = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{fm}\n---\n\n{body}"


# -- operations ------------------------------------------------------------

@skill
def read_note(path: str) -> Result:
    """Read a note. Returns {'path', 'frontmatter', 'body', 'raw'}."""
    target = _resolve(path)
    if not target.exists():
        return Result.failure(f"note not found: {path}")
    raw = target.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw)
    return Result.success(
        {"path": _relativize(target), "frontmatter": frontmatter, "body": body, "raw": raw}
    )


@skill
def write_note(
    path: str,
    body: str,
    frontmatter: dict | None = None,
    overwrite: bool = False,
) -> Result:
    """Create a note, making parent folders as needed.

    Refuses to overwrite an existing note unless ``overwrite=True`` -- protects
    the existing vault from an agent clobbering real content. Stamps a stable
    ``uid`` (generated once, preserved across every rewrite/rename/move),
    ``created`` (first write), and ``updated`` automatically -- so every agent
    note is consistently identifiable and Axiom-organizable.
    """
    target = _resolve(path)
    existed = target.exists()
    if existed and not overwrite:
        return Result.failure(f"note already exists (pass overwrite=True): {path}")

    meta = dict(frontmatter or {})
    today = date.today().isoformat()
    uid = meta.pop("uid", None)
    if existed:
        # Carry the stable uid + created forward so an overwrite never wipes them.
        try:
            old_meta, _ = _split_frontmatter(target.read_text(encoding="utf-8"))
            uid = uid or old_meta.get("uid")
            if "created" not in meta and old_meta.get("created"):
                meta["created"] = old_meta["created"]
        except Exception:
            pass
    if not existed:
        meta.setdefault("created", today)
    # uid leads the frontmatter; generated once if the note has never had one.
    meta = {"uid": uid or new_uid(), **meta, "updated": today}

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_compose(meta, body), encoding="utf-8")
    log.info("wrote note %s%s", _relativize(target), " (overwrote)" if existed else "")
    return Result.success({"path": _relativize(target), "overwritten": existed})


@skill
def ensure_uids(subfolder: str = "") -> Result:
    """Backfill a stable ``uid`` onto every note that lacks one. Idempotent.

    Non-destructive: only prepends a uid to the frontmatter, leaving the body and
    all existing fields untouched. Safe to run on the whole vault (incl. School/).
    """
    root = _resolve(subfolder, as_dir=True) if subfolder else config.vault_path.resolve()
    scanned = added = 0
    for p in root.rglob("*.md"):
        if any(part.startswith(".") for part in p.relative_to(config.vault_path).parts):
            continue  # skip .obsidian/.trash etc.
        scanned += 1
        try:
            meta, body = _split_frontmatter(p.read_text(encoding="utf-8"))
            if meta.get("uid"):
                continue
            p.write_text(_compose({"uid": new_uid(), **meta}, body), encoding="utf-8")
            added += 1
        except Exception:
            continue
    log.info("ensure_uids: %d/%d notes got a new uid", added, scanned)
    return Result.success({"scanned": scanned, "added": added})


@skill
def append_note(path: str, text: str, create: bool = True) -> Result:
    """Append text to a note (for logs, running lists). Creates it if missing.

    Appends raw text after the existing content -- does not touch frontmatter.
    """
    target = _resolve(path)
    if not target.exists():
        if not create:
            return Result.failure(f"note not found and create=False: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
        log.info("created note via append %s", _relativize(target))
        return Result.success({"path": _relativize(target), "created": True})

    existing = target.read_text(encoding="utf-8")
    sep = "" if existing == "" or existing.endswith("\n") else "\n"
    with target.open("a", encoding="utf-8") as fh:
        fh.write(sep + text.rstrip("\n") + "\n")
    log.info("appended to note %s", _relativize(target))
    return Result.success({"path": _relativize(target), "created": False})


@skill
def search(query: str, folder: str | None = None, limit: int = 20) -> Result:
    """Case-insensitive search over note filenames and content.

    Scope to a subfolder with ``folder``. Returns up to ``limit`` hits, each
    {'path', 'matched_in', 'snippet'}.
    """
    vault = config.vault_path.resolve()
    root = _resolve(folder, as_dir=True) if folder else vault
    if folder and not root.is_dir():
        return Result.failure(f"folder not found: {folder}")

    needle = query.lower()
    hits: list[dict] = []
    for md in sorted(root.rglob("*.md")):
        if ".obsidian" in md.parts:
            continue
        name_hit = needle in md.stem.lower()
        snippet = None
        matched_in = "filename" if name_hit else None
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        low = text.lower()
        if needle in low:
            matched_in = "filename+content" if name_hit else "content"
            idx = low.find(needle)
            start = max(0, idx - 40)
            snippet = text[start:idx + len(query) + 40].replace("\n", " ").strip()
        if matched_in:
            hits.append({"path": _relativize(md), "matched_in": matched_in, "snippet": snippet})
        if len(hits) >= limit:
            break
    return Result.success(hits)


@skill
def list_notes(folder: str | None = None) -> Result:
    """List vault-relative paths of all notes, optionally within a folder."""
    vault = config.vault_path.resolve()
    root = _resolve(folder, as_dir=True) if folder else vault
    if folder and not root.is_dir():
        return Result.failure(f"folder not found: {folder}")
    notes = [
        _relativize(md)
        for md in sorted(root.rglob("*.md"))
        if ".obsidian" not in md.parts
    ]
    return Result.success(notes)


@skill
def move_note(src: str, dst: str, overwrite: bool = False) -> Result:
    """Move a note within the vault (e.g. into ``_Archive/``). Confined to the vault.

    Creates the destination's parent folders. Refuses to clobber an existing
    destination unless ``overwrite=True``. This is how the librarian *archives* a
    note instead of deleting it -- the file is preserved, just relocated. Obsidian
    resolves ``[[wikilinks]]`` by basename, so links to a moved note still work.
    """
    source = _resolve(src)
    if not source.exists():
        return Result.failure(f"note not found: {src}")
    target = _resolve(dst)
    if target.exists() and not overwrite:
        return Result.failure(f"destination exists (pass overwrite=True): {dst}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
    log.info("moved note %s -> %s", src, _relativize(target))
    return Result.success({"from": src, "to": _relativize(target)})


def note_exists(path: str) -> bool:
    """Quick existence check (not a Result -- can't meaningfully fail)."""
    try:
        return _resolve(path).exists()
    except ValueError:
        return False
