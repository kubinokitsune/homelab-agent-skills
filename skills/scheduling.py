"""Tiny in-process scheduler -- agents run their own daily jobs.

No OS cron or Task Scheduler (which needs admin on Windows). Since each agent
already runs 24/7 under its autostart wrapper, it can schedule its own work:

    import asyncio
    from skills.scheduling import run_daily

    asyncio.create_task(run_daily(2, 0, nightly_job, name="axiom-nightly", logger=log))

``run_daily`` sleeps until the next HH:MM (local time), runs the job, and
repeats forever. A job that raises is logged and the loop continues.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Awaitable, Callable


async def run_daily(
    hour: int,
    minute: int,
    job: Callable[[], Awaitable[None]],
    *,
    name: str = "job",
    logger=None,
) -> None:
    """Run ``job`` every day at ``hour:minute`` local time, forever."""
    while True:
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())
        try:
            await job()
        except Exception:
            if logger is not None:
                logger.exception("daily job %r failed", name)
