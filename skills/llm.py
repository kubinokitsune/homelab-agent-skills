"""Synchronous local-LLM chat -- the reasoning half, mirror of vector_store's embed.

``vector_store._embed`` turns text into vectors via Ollama; this turns a prompt
into a text completion via Ollama's ``/api/chat``. Synchronous and
dependency-light (urllib) so blocking skills -- run off the event loop in a
thread -- can reason without pulling in aiohttp.

``DiscordAgent.ask`` stays the async path for live Discord replies; this is for
background jobs like Axiom's nightly librarian pass.

    reply = llm.chat("Are these two notes redundant? Answer yes or no.",
                     system="You organize notes. Be strict.")
"""

from __future__ import annotations

import json
import urllib.request

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)

_CHAT_TIMEOUT = 180  # local CPU inference can be slow; give it room


@skill
def chat(prompt: str, system: str = "", model: str | None = None, temperature: float = 0.2) -> Result:
    """One-shot chat completion from the local model. Returns the reply text.

    Low default temperature -- these calls are for judgment/clustering, where we
    want consistency, not creativity.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = json.dumps({
        "model": model or config.default_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()
    req = urllib.request.Request(
        f"{config.ollama_host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_CHAT_TIMEOUT) as resp:
        data = json.loads(resp.read())
    return Result.success(data["message"]["content"])
