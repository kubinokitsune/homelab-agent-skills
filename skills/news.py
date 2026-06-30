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
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)

_UA = "homelab-iris/0.1 (+https://github.com/kubinokitsune; personal news reader)"

# All RSS/Atom, no API key needed. Reddit's JSON endpoint is blocked now, but
# its .rss feeds work. Tweak freely.
#
# World + Costa Rica news (general headlines) -- listed first so they lead the
# digest. NOTE: CNN's public RSS was intentionally left out -- it now serves
# stale cached articles (old headlines), which would be misleading in a daily
# briefing. NYT + BBC cover international; La Nacion + Tico Times cover Costa
# Rica; El Pais adds Spanish-language world news.
NEWS_RSS = {
    "NYT World": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "La Nación (CR)": "https://www.nacion.com/arc/outboundfeeds/rss/?outputType=xml",
    "Tico Times (CR)": "https://ticotimes.net/feed",
    "El País": "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",
}
# Maker / tech / hardware feeds.
TECH_RSS = {
    "Hackaday": "https://hackaday.com/blog/feed/",
    "Tom's Hardware": "https://www.tomshardware.com/feeds/all",
    "Hacker News": "https://hnrss.org/frontpage",
    "Hackster": "https://www.hackster.io/projects?format=atom",
}
# Combined feed order: real news first, then tech. headlines() iterates this.
DEFAULT_RSS = {**NEWS_RSS, **TECH_RSS}
# Reddit subs -> their .rss feeds (fetched with spacing to avoid 429).
DEFAULT_REDDIT = ["homelab", "PrintedCircuitBoard", "lacrosse"]


def _get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# Anything published before this many days ago is treated as stale and dropped.
# Guards against feeds that serve old cached articles (the reason CNN was cut).
MAX_AGE_DAYS = 366  # ~1 year


def _local(tag: str) -> str:
    return tag.split("}")[-1]  # strip XML namespace


def _parse_date(s: str | None) -> datetime | None:
    """Parse an RSS (RFC 822) or Atom (ISO 8601) date into an aware datetime."""
    if not s:
        return None
    s = s.strip()
    try:  # RSS 2.0: "Mon, 29 Jun 2026 02:00:09 +0000" / "... GMT"
        dt = parsedate_to_datetime(s)
        if dt:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    try:  # Atom: "2026-06-13T05:09:52Z"
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _rss(url: str, limit: int, max_age_days: int = MAX_AGE_DAYS) -> list[dict]:
    """Parse RSS 2.0 or Atom into [{title, url, date}], newest-first.

    Items older than ``max_age_days`` are skipped so stale feeds (old cached
    articles) never reach the digest. Items with no parseable date are kept.
    """
    root = ET.fromstring(_get(url))
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    out: list[dict] = []
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        title, link, pub = None, None, None
        for child in el:
            t = _local(child.tag)
            if t == "title" and child.text:
                title = child.text.strip()
            elif t == "link":
                link = child.get("href") or (child.text or "").strip()  # atom uses href
            elif t in ("pubDate", "published", "updated", "date") and child.text and not pub:
                pub = child.text.strip()
        if not title:
            continue
        dt = _parse_date(pub)
        if dt and dt < cutoff:
            continue  # too old -- skip stale item
        out.append({"title": title, "url": link or "",
                    "date": dt.date().isoformat() if dt else None})
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
