"""The single return type every skill hands back to an agent.

Skills in this library don't raise on expected failures (a Discord 503, a
missing vault note, a Qdrant timeout) and they don't return bare booleans that
throw away the reason. They return a ``Result``: one predictable shape the
calling agent branches on.

    result = some_skill(...)
    if result.ok:
        use(result.data)
    else:
        log.warning(result.error)   # always a human-readable string on failure

Why this matters here specifically: these agents run unattended overnight. A
skill that raises an unhandled exception at 3am can take the whole agent down
silently. A skill that returns ``Result.failure("Discord returned 503")`` lets
the agent log it, fall through to a backup channel, and keep running.

Deliberately tiny -- three fields, no error codes or severity. Severity is the
notification layer's job, not the return type's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Result:
    """Outcome of a skill call.

    On success: ``ok`` is True, ``data`` holds the payload, ``error`` is None.
    On failure: ``ok`` is False, ``data`` is None, ``error`` is a human string.

    Construct via the ``success`` / ``failure`` classmethods rather than the
    raw constructor, so intent is obvious at the call site.
    """

    ok: bool
    data: Any = None
    error: str | None = None

    @classmethod
    def success(cls, data: Any = None) -> "Result":
        """A successful outcome, optionally carrying a payload."""
        return cls(ok=True, data=data, error=None)

    @classmethod
    def failure(cls, error: str) -> "Result":
        """A failed outcome carrying a human-readable reason."""
        return cls(ok=False, data=None, error=error)

    def __bool__(self) -> bool:
        """Allow ``if result:`` as shorthand for ``if result.ok:``."""
        return self.ok
