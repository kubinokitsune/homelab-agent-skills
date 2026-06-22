"""News & feeds for Iris's digest -- the no-key sources from the master doc.

RSS (open XML) and Reddit (public JSON endpoint) need no API keys, just a real
User-Agent. Returns headlines grouped by source. Dependency-free (stdlib only).

    news.headlines(per_source=3)   # [{source, items:[{title,url}]}, ...]

GitHub releases / Reddit OAuth / LCSC can be layered on later with free keys.
"""

from __future__ import annotations

import time
import urllib.request
import xml.etree.ElementTree as ET

from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)

_UA = "homelab-iris/0.1 (+https://github.com/kubinokitsune; personal news reader)"

# Master-doc sources + a couple maker feeds, all RSS/Atom (no API key needed).
# Reddit's JSON endpoint is blocked now, but its .rss feeds work. Tweak freely.
DEFAULT_RSS = {
    "Hackaday": "https://hackaday.com/blog/feed/",
    "Tom's Hardware": "https://www.tomshardware.com/feeds/all",
    "Hacker News": "https://hnrss.org/frontpage",
    "Hackster": "https://www.hackster.io/projects?format=atom",
}
# Reddit subs -> their .rss feeds (fetched with spacing to avoid 429).
DEFAULT_REDDIT = ["homelab", "PrintedCircuitBoard", "lacrosse"]


def _get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _local(tag: str) -> str:
    return tag.split("}")[-1]  # strip XML namespace


def _rss(url: str, limit: int) -> list[dict]:
    """Parse RSS 2.0 or Atom into [{title, url}]. Tolerant of either format."""
    root = ET.fromstring(_get(url))
    out: list[dict] = []
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        title, link = None, None
        for child in el:
            t = _local(child.tag)
            if t == "title" and child.text:
                title = child.text.strip()
            elif t == "link":
                link = child.get("href") or (child.text or "").strip()  # atom uses href
        if title:
            out.append({"title": title, "url": link or ""})
        if len(out) >= limit:
            break
    return out


@skill
def headlines(per_source: int = 3) -> Result:
    """Aggregate headlines from the configured RSS + Reddit feeds, grouped.

    Each source is fetched independently; one failing source never sinks the rest.
    Reddit feeds are spaced out to stay under its rate limit.
    """
    groups: list[dict] = []
    for name, url in DEFAULT_RSS.items():
        try:
            items = _rss(url, per_source)
        except Exception as exc:
            log.warning("RSS '%s' failed: %s", name, exc)
            items = []
        if items:
            groups.append({"source": name, "items": items})
    for i, sub in enumerate(DEFAULT_REDDIT):
        if i:
            time.sleep(1.2)  # space reddit requests to avoid 429
        try:
            items = _rss(f"https://www.reddit.com/r/{sub}/top.rss?t=week", per_source)
        except Exception as exc:
            log.warning("reddit r/%s failed: %s", sub, exc)
            items = []
        if items:
            groups.append({"source": f"r/{sub}", "items": items})
    return Result.success(groups)
