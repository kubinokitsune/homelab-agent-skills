"""Server filesystem operations -- confined to an allowlist of roots.

Axiom uses this to catalogue STL/G-code/project files; Hermes uses it for disk
checks and log cleanup. Unlike the vault skill (one fixed folder, relative
paths), file_ops roams the server, so safety is an explicit **allowlist**:

* Every path is absolute and must resolve inside one of ``config.file_ops_roots``
  (env ``FILE_OPS_ROOTS``). Anything outside -> clean failure. With no roots
  configured, every operation refuses -- nothing roams by accident.
* Writes never clobber unless ``overwrite=True``.
* Reads have a size guard so an agent can't slurp a giant file into memory.
* ``delete_file`` handles single files only -- never directories, never
  recursive. (Honors the "never auto-delete broadly" principle from the doc.)
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)

_DEFAULT_MAX_READ = 1_000_000  # 1 MB guard on read_file


def _resolve_allowed(path: str) -> Path:
    """Resolve an absolute path, confined to the configured allowlist roots."""
    roots = config.file_ops_roots
    if not roots:
        raise ValueError(
            "no FILE_OPS_ROOTS configured -- set it in .env before using file_ops"
        )
    candidate = Path(path).resolve()
    for root in roots:
        if candidate == root or root in candidate.parents:
            return candidate
    raise ValueError(f"path '{path}' is outside all allowed FILE_OPS_ROOTS")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _describe(p: Path) -> dict:
    st = p.stat()
    return {
        "path": str(p),
        "name": p.name,
        "type": "dir" if p.is_dir() else "file",
        "size": st.st_size,
        "modified": _iso(st.st_mtime),
    }


# -- operations ------------------------------------------------------------

@skill
def list_dir(path: str, recursive: bool = False) -> Result:
    """List a directory's contents. Each entry has name/path/type/size/modified."""
    target = _resolve_allowed(path)
    if not target.is_dir():
        return Result.failure(f"not a directory: {path}")
    items = target.rglob("*") if recursive else target.iterdir()
    entries = [_describe(p) for p in sorted(items)]
    return Result.success(entries)


@skill
def read_file(path: str, max_bytes: int = _DEFAULT_MAX_READ) -> Result:
    """Read a text file. Refuses files larger than ``max_bytes`` (raise it to override)."""
    target = _resolve_allowed(path)
    if not target.is_file():
        return Result.failure(f"not a file: {path}")
    size = target.stat().st_size
    if size > max_bytes:
        return Result.failure(
            f"file is {size} bytes (> max_bytes={max_bytes}); raise max_bytes to read it"
        )
    text = target.read_text(encoding="utf-8", errors="replace")
    return Result.success({"path": str(target), "size": size, "text": text})


@skill
def write_file(path: str, content: str, overwrite: bool = False) -> Result:
    """Write a text file, creating parent dirs. Refuses to clobber unless overwrite."""
    target = _resolve_allowed(path)
    existed = target.exists()
    if existed and not overwrite:
        return Result.failure(f"file already exists (pass overwrite=True): {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    log.info("wrote file %s%s", target, " (overwrote)" if existed else "")
    return Result.success({"path": str(target), "overwritten": existed})


@skill
def find_files(root: str, pattern: str = "*", older_than_days: float | None = None) -> Result:
    """Find files under ``root`` matching a glob ``pattern``.

    With ``older_than_days`` set, returns only files not modified in that many
    days -- e.g. Axiom flagging files untouched for 90+ days.
    """
    base = _resolve_allowed(root)
    if not base.is_dir():
        return Result.failure(f"not a directory: {root}")
    cutoff = time.time() - older_than_days * 86400 if older_than_days else None
    hits = []
    for p in sorted(base.rglob(pattern)):
        if not p.is_file():
            continue
        if cutoff is not None and p.stat().st_mtime > cutoff:
            continue
        hits.append(_describe(p))
    return Result.success(hits)


@skill
def file_info(path: str) -> Result:
    """Stat a file or directory: type, size, modified, created."""
    target = _resolve_allowed(path)
    if not target.exists():
        return Result.failure(f"not found: {path}")
    st = target.stat()
    return Result.success({
        "path": str(target),
        "type": "dir" if target.is_dir() else "file",
        "size": st.st_size,
        "modified": _iso(st.st_mtime),
        "created": _iso(st.st_ctime),
    })


@skill
def delete_file(path: str) -> Result:
    """Delete a single file. Refuses directories -- never recursive."""
    target = _resolve_allowed(path)
    if not target.exists():
        return Result.failure(f"not found: {path}")
    if target.is_dir():
        return Result.failure(f"refusing to delete a directory: {path}")
    target.unlink()
    log.info("deleted file %s", target)
    return Result.success({"path": str(target)})
