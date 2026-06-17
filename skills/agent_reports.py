"""Daily report handoff -- how agents feed Iris.

Each agent writes a short markdown report once a day; Iris reads them all and
compiles the digest. Reports live under ``config.agent_reports_dir`` as
``{date}/{Agent}.md`` -- a plain folder outside the vault, so daily machine
output never clutters Pipe's notes or skews Axiom's index.

    agent_reports.write_report("Axiom", report_markdown)
    reports = agent_reports.read_reports()       # {"Axiom": "...", "Forge": "..."}
"""

from __future__ import annotations

from datetime import date

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)


def _today() -> str:
    return date.today().isoformat()


@skill
def write_report(agent: str, content: str, day: str | None = None) -> Result:
    """Write (overwrite) an agent's report for a day. ``day`` defaults to today."""
    day = day or _today()
    folder = config.agent_reports_dir / day
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{agent}.md"
    path.write_text(content, encoding="utf-8")
    log.info("wrote %s report for %s", agent, day)
    return Result.success({"path": str(path), "day": day})


@skill
def read_reports(day: str | None = None) -> Result:
    """Return all agent reports for a day as ``{agent_name: markdown}``."""
    day = day or _today()
    folder = config.agent_reports_dir / day
    if not folder.is_dir():
        return Result.success({})
    reports = {p.stem: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.md"))}
    return Result.success(reports)
