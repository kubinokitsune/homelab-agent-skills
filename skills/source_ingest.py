"""Source ingestion -- the NotebookLM-style pipeline behind Codex.

Takes a source's text (from a dropped file, a pasted URL, or pasted text),
classifies what it's about by nearest-neighbour against the existing vault, and:

  * writes a **source note** (the cleaned content) and a **synthesis note**
    (LLM summary + key points) into the right place,
  * routes engineering sources near the engineering projects and school sources
    under the relevant class,
  * links both notes to the domain node so Obsidian connects them,
  * embeds both into ``vault_index`` (everyone, incl. Chiron) and, for
    engineering, ``forge_memory`` (so Forge cites them). School notes are plain
    markdown under ``School/`` where Athena reads them.

So both the uploaded content AND the notes Codex makes from it become sources the
other agents can use.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from skills import llm
from skills import obsidian_vault as vault
from skills import vault_index
from skills import vector_store as vs
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)

INDEX = "vault_index"
SOURCES = "vault_sources"  # ingested/filed sources only -- Codex's NotebookLM corpus
TEXT_EXTS = {"txt", "md", "markdown", "text", "csv", "log"}
ENG_PREFIXES = ("Engineering Studies/", "04_Projects/")
MIN_SIGNAL = 0.55        # nearest-neighbour score below which we treat as generic
                         # (real-domain content scores 0.65+; off-topic ~0.48)
ENG_SOURCES = "Engineering Studies/Projects/Sources"
REF_SOURCES = "05_Reference/Sources"
# Real class subfolders under School/ -- preferred over generic ones like
# "Assignments" when routing a school source, so sources land "by class".
SCHOOL_CLASSES = {"math", "english", "spanish", "physics", "chemistry",
                  "biology", "history", "economics", "tok"}


def _safe_title(title: str) -> str:
    title = re.sub(r"[\\/:*?\"<>|]+", " ", title).strip()
    return (title[:80] or f"Source {datetime.now():%Y%m%d-%H%M%S}").strip()


def _subject_of(doc_id: str) -> str:
    parts = doc_id.split("/")
    return parts[1] if len(parts) > 1 else "General"


def _class_in_path(doc_id: str) -> str | None:
    """A known class found anywhere in the path (the vault nests classes under
    e.g. School/Assignments/Spanish/...), or None."""
    for part in doc_id.split("/"):
        if part.lower() in SCHOOL_CLASSES:
            return part
    return None


def _classify(text: str) -> dict:
    """Decide domain + target folder + the node to link, by nearest neighbours."""
    res = vs.query(INDEX, text, 8)
    hits = [h for h in (res.data or []) if (h.get("score") or 0) >= MIN_SIGNAL]
    eng = [h for h in hits if str(h["doc_id"]).startswith(ENG_PREFIXES)]
    school = [h for h in hits if str(h["doc_id"]).startswith("School/")]
    best_eng = max((h["score"] for h in eng), default=0.0)
    best_school = max((h["score"] for h in school), default=0.0)

    if best_eng and best_eng >= best_school:
        proj = next((h["doc_id"] for h in eng
                     if str(h["doc_id"]).startswith("Engineering Studies/Projects/")), None)
        link = Path(proj).stem if proj else "Engineering Projects MOC"
        return {"domain": "engineering", "folder": ENG_SOURCES, "link": link,
                "extra_collection": "forge_memory"}
    if best_school:
        # Find a real class anywhere in a school hit's path (sources land "by class");
        # fall back to the top-level folder if no class is recognized.
        subject = next((_class_in_path(h["doc_id"]) for h in school
                        if _class_in_path(h["doc_id"])), None)
        if not subject:
            subject = _subject_of(school[0]["doc_id"])
        return {"domain": "school", "subject": subject,
                "folder": f"School/{subject}/Sources", "link": subject}
    return {"domain": "reference", "folder": REF_SOURCES, "link": None}


_PROJ_STOP = {"and", "the", "for", "in", "of", "to", "a", "an"}


def _project_match(hint: str) -> tuple[str | None, int]:
    """Best-matching engineering project note for a hint, by name-token overlap.
    Returns (project_note_stem, score) where score = distinct project-name words
    found in the hint. Lets Pipe say "F1", "chemistry calculator", "3D printer"."""
    low = hint.lower()
    listed = vault.list_notes(folder="Engineering Studies/Projects")
    best, best_score = None, 0
    for rel in (listed.data or []):
        if "/Sources/" in rel:
            continue
        stem = Path(rel).stem
        toks = {t for t in re.split(r"[^a-z0-9]+", stem.lower())
                if len(t) > 1 and t not in _PROJ_STOP}
        matched = {t for t in toks if re.search(rf"\b{re.escape(t)}\b", low)}
        if len(matched) > best_score:
            best, best_score = stem, len(matched)
    return best, best_score


def _hint_route(hint: str) -> dict | None:
    """If the caption names a class or a specific project, route by that instead of
    guessing. Lets Pipe steer a source whose *intended* use differs from its
    content (a PLA paper that reads like '3D printing' but is for his Physics IA),
    or target a project directly ("for the F1 car")."""
    if not hint:
        return None
    low = hint.lower()
    proj, pscore = _project_match(hint)

    def engineering(link):
        return {"domain": "engineering", "folder": ENG_SOURCES, "link": link,
                "extra_collection": "forge_memory"}

    # Strong, multi-word project name ("chemistry calculator", "3D printer").
    if pscore >= 2:
        return engineering(proj)
    # A class word ("physics", "math", "chemistry").
    for cls in SCHOOL_CLASSES:
        if re.search(rf"\b{cls}\b", low):
            subj = cls.upper() if cls == "tok" else cls.capitalize()
            return {"domain": "school", "subject": subj,
                    "folder": f"School/{subj}/Sources", "link": subj}
    # A weaker single-word project hint ("F1", "ender") -- only after class check.
    if pscore >= 1:
        return engineering(proj)
    if re.search(r"\b(engineering|forge|project|pcb|cad|hardware|firmware)\b", low):
        return engineering("Engineering Projects MOC")
    if re.search(r"\b(reference|misc|general|other)\b", low):
        return {"domain": "reference", "folder": REF_SOURCES, "link": None}
    return None


def _summarize(title: str, text: str, model: str | None) -> str:
    prompt = (
        f"Summarize this source titled '{title}' for a study/reference note. Give:\n"
        "1. A 2-3 sentence overview.\n2. 4-6 key points as bullets.\n"
        "3. A short 'Key terms' list if technical.\n"
        "Be faithful to the source; don't invent. Markdown, no preamble.\n\n"
        f"--- SOURCE ---\n{text[:6000]}"
    )
    res = llm.chat(prompt, "You write tight, faithful study summaries.", model)
    return res.data.strip() if res.ok else "_(summary unavailable -- model error)_"


@skill
def ingest_text(title: str, text: str, origin: str = "paste",
                model: str | None = None, hint: str = "", summarize: bool = True) -> Result:
    """Ingest one source's text: classify, write a source note (and, if
    ``summarize``, a synthesis note), embed, link. Returns a dict of what was
    created and where. A ``hint`` (e.g. a drop caption like "physics IA")
    overrides the content-based routing. ``summarize=False`` is the filing mode
    (Axiom): just sort the document into place, no summary."""
    text = (text or "").strip()
    if len(text) < 40:
        return Result.failure("source text too short to ingest")
    title = _safe_title(title)
    cls = _hint_route(hint) or _classify(text)
    folder = cls["folder"]
    link = cls.get("link")
    stamp = datetime.now().strftime("%Y-%m-%d")
    # vault_index = everyone; vault_sources = Codex's source-only corpus;
    # forge_memory = engineering, so Forge cites them.
    collections = [INDEX, SOURCES] + ([cls["extra_collection"]] if cls.get("extra_collection") else [])
    link_line = f"\n\n_Related: [[{link}]]_\n" if link else ""

    # Source note (the cleaned content).
    src_path = f"{folder}/{title}.md"
    src_body = f"> [!quote] Ingested {stamp} (origin: {origin})\n\n{text}{link_line}"
    w1 = vault.write_note(src_path, src_body,
                          {"type": "source", "origin": origin,
                           "tags": [cls["domain"], "source"]}, overwrite=True)
    if not w1.ok:
        return Result.failure(f"couldn't write source note: {w1.error}")
    for coll in collections:
        vault_index.index_one(coll, src_path)

    # Optional synthesis note (the agent's own notes from the source).
    summary, syn_path = "", None
    if summarize:
        summary = _summarize(title, text, model)
        syn_path = f"{folder}/{title} -- Summary.md"
        syn_body = f"_Synthesis of [[{title}]] ({stamp})._\n\n{summary}{link_line}"
        vault.write_note(syn_path, syn_body,
                         {"type": "source-summary", "tags": [cls["domain"], "source", "summary"]},
                         overwrite=True)
        for coll in collections:
            vault_index.index_one(coll, syn_path)

    log.info("ingested '%s' -> %s (domain=%s, link=%s, summary=%s)",
             title, folder, cls["domain"], link, summarize)
    return Result.success({
        "title": title, "domain": cls["domain"], "subject": cls.get("subject"),
        "link": link, "source_path": src_path, "summary_path": syn_path, "summary": summary,
    })


def extract_file(filename: str, data: bytes) -> tuple[str, str] | None:
    """(title, text) from a dropped file's bytes -- PDF or text. None if the type
    isn't supported. Shared by Codex and Axiom's file intake."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    title = Path(filename).stem
    if ext == "pdf":
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return title, "\n\n".join((p.extract_text() or "") for p in reader.pages).strip()
    if ext in TEXT_EXTS:
        return title, data.decode("utf-8", errors="ignore")
    return None


# -- extractors ------------------------------------------------------------

def extract_pdf(path: str) -> str:
    """Text from a PDF via pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n\n".join((p.extract_text() or "") for p in reader.pages).strip()


def extract_url(url: str) -> tuple[str, str]:
    """(title, text) from a web page -- crude HTML strip, dependency-free."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (homelab Codex)"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else url
    html = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&[a-z#0-9]+;", " ", text)
    text = re.sub(r"\s+\n", "\n", re.sub(r"[ \t]+", " ", text))
    return title, text.strip()
