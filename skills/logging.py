"""One logging setup, shared by every skill and agent.

A skill never configures logging itself. It just asks for a logger:

    from skills.logging import get_logger
    log = get_logger(__name__)
    log.info("queued print job %s", job_id)

Every line carries three things so you can trace it later:

    2026-06-16 16:15:01  INFO   [Forge]  homelab.skills.notifications  queued ...
    └── date + time      └─level └─agent  └─ which skill                └─ message

* **date + time** -- when it happened (``%(asctime)s``)
* **agent** -- which agent process emitted it (Forge, Mason, ...). Set once at
  startup with ``set_agent("Forge")`` or via the ``AGENT_NAME`` env var; defaults
  to ``system`` for standalone skill runs.
* **skill** -- the module that logged it (``homelab.skills.notifications``)

Every logger nests under a single ``homelab`` root, so they all share one
handler and one format. Configuration happens once, lazily, the first time any
logger is requested -- call it from a hundred skills and you still get exactly
one handler (no duplicated log lines).

Design choices for this stack:

* Logs go to **stdout**, not files. In LXC containers, Docker/Loki capture
  stdout automatically; writing our own files would fight that model.
* The format lives in exactly **one place** (``_build_handler``). When Loki
  arrives and you want JSON, you change that one function -- nothing else.
* Level comes from ``config.log_level`` (env ``LOG_LEVEL``).
"""

from __future__ import annotations

import logging
import sys

from skills.config import config

_ROOT = "homelab"
_configured = False

# Set once per process by the running agent; falls back to config.agent_name.
_agent_name: str | None = None


def set_agent(name: str) -> None:
    """Declare which agent this process is, so every log line is tagged.

    Call this once at agent startup, before logging anything:

        from skills.logging import set_agent
        set_agent("Forge")
    """
    global _agent_name
    _agent_name = name


def _current_agent() -> str:
    return _agent_name or config.agent_name


class _AgentFilter(logging.Filter):
    """Inject the current agent name onto every record as ``record.agent``."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.agent = _current_agent()
        return True


def _build_handler() -> logging.Handler:
    """Build the single stdout handler. The one place the format is defined.

    To switch to JSON for Loki later, swap the formatter here and nothing else
    in the codebase changes.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_AgentFilter())
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-7s  [%(agent)s]  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


def _configure() -> None:
    """Attach the handler to the homelab root logger exactly once."""
    global _configured
    if _configured:
        return
    root = logging.getLogger(_ROOT)
    root.addHandler(_build_handler())
    root.setLevel(config.log_level)
    root.propagate = False  # don't double-log through Python's root logger
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger nested under the homelab root.

    Pass ``__name__`` from the calling module so the source shows up in every
    line, e.g. ``homelab.skills.notifications``.
    """
    _configure()
    return logging.getLogger(f"{_ROOT}.{name}")
