"""The error-handling convention, made reusable.

The contract for every I/O skill in this library:

    A skill catches its own exceptions, LOGS them (with traceback), and returns
    Result.failure(...). An exception never escapes a skill to crash the agent.

Writing that try/except by hand in every skill is repetitive and drifts out of
sync. The ``@skill`` decorator below does it once: wrap a function whose happy
path returns a ``Result``, and any exception it raises is automatically logged
with a full traceback and converted to ``Result.failure(...)``.

    from skills.errors import skill
    from skills.result import Result

    @skill
    def send_discord(channel: str, message: str) -> Result:
        ...                       # just the happy path
        return Result.success(message_id)

    # If anything inside raises, the caller transparently gets:
    #   Result(ok=False, data=None, error="send_discord failed: <reason>")

The decorator is the sensible default, not a mandate. A skill that wants a
tailored error message per failure mode can still write its own try/except and
return ``Result.failure(...)`` directly -- the decorator just removes the
boilerplate for the 90% case.
"""

from __future__ import annotations

import functools
from typing import Callable

from skills.logging import get_logger
from skills.result import Result


def skill(func: Callable[..., Result]) -> Callable[..., Result]:
    """Wrap a Result-returning skill so exceptions become Result.failure.

    The wrapped function should return a ``Result`` on its happy path. If it
    raises instead, the exception is logged (with traceback) under the skill's
    own module logger, and a ``Result.failure`` carrying a clean reason string
    is returned to the caller.
    """

    log = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Result:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- deliberate catch-all boundary
            log.exception("%s failed", func.__name__)
            return Result.failure(f"{func.__name__} failed: {exc}")

    return wrapper
