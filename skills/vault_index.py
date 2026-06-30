"""Embed vault notes into a Qdrant collection -- the shared indexer.

Any agent that grounds itself in vault content uses this: Forge (engineering
folders -> forge_memory), Scout (recruiting folders -> scout_memory), Axiom
(the whole vault -> vault_index). Re-running is safe -- each note upserts under
its own path, so content updates in place instead of duplicating.
"""

from __future__ import annotations

from pathlib import Path

from skills import obsidian_vault as vault
from skills import vector_store as vs
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)

_MAX_EMBED_CHARS = 6000  # keep within the embedder's context window


@skill
def index(collection: str, folders: list[str] | None = None) -> Result:
    """Embed notes into ``collection``. Returns Result with the count indexed.

    If ``folders`` is None, indexes the **whole vault**; otherwise just the named
    top-level folders. Empty notes are skipped. Each note's payload carries its
    title and folder so query results are self-describing.
    """
    if folders is None:
        listings = [("", vault.list_notes())]
    else:
        listings = [(f, vault.list_notes(folder=f)) for f in folders]

    indexed = 0
    for folder, listed in listings:
        if not listed.ok:
            log.warning("skipping folder '%s': %s", folder or "(vault)", listed.error)
            continue
        for rel_path in listed.data:
            # Archived notes are kept for recovery but shouldn't show in search.
            if rel_path.startswith("_Archive/"):
                continue
            note = vault.read_note(rel_path)
            if not note.ok:
                continue
            body = note.data["body"].strip()
            if not body:
                continue
            title = Path(rel_path).stem
            top = folder or (rel_path.split("/")[0] if "/" in rel_path else "(root)")
            text = f"{title}\n\n{body}"[:_MAX_EMBED_CHARS]
            r = vs.upsert(collection, rel_path, text,
                          payload={"title": title, "folder": top,
                                   "uid": note.data["frontmatter"].get("uid")})
            if r.ok:
                indexed += 1
    log.info("indexed %d notes into '%s'", indexed, collection)
    return Result.success({"indexed": indexed, "collection": collection})


@skill
def index_one(collection: str, rel_path: str) -> Result:
    """Embed a single note into ``collection`` -- the incremental path.

    Used when an agent is told a specific note changed (e.g. Axiom hears from
    Forge that a note was saved) and wants to fold just that note in without
    re-embedding the whole vault. Uses the same doc_id/payload schema as
    ``index`` so query results stay consistent. Empty notes are skipped.
    """
    note = vault.read_note(rel_path)
    if not note.ok:
        return Result.failure(f"couldn't read note '{rel_path}': {note.error}")
    body = note.data["body"].strip()
    if not body:
        return Result.failure(f"note '{rel_path}' is empty -- nothing to index")
    title = Path(rel_path).stem
    top = rel_path.split("/")[0] if "/" in rel_path else "(root)"
    payload = {"title": title, "folder": top, "uid": note.data["frontmatter"].get("uid")}
    text = f"{title}\n\n{body}"[:_MAX_EMBED_CHARS]
    r = vs.upsert(collection, rel_path, text, payload=payload)
    if not r.ok:
        # Dense text can exceed the embedder's context window (Ollama 500s).
        # Retry on a shorter prefix so the note still lands in the index.
        log.warning("index_one '%s' failed (%s); retrying shorter", rel_path, r.error)
        r = vs.upsert(collection, rel_path, text[:1500], payload=payload)
        if not r.ok:
            return Result.failure(r.error)
    log.info("indexed one note '%s' into '%s'", rel_path, collection)
    return Result.success({"indexed": 1, "doc_id": rel_path, "collection": collection})
