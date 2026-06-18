"""Reusable Discord-agent scaffolding shared by every conversational agent.

Each agent (Forge, Scout, Mason, ...) is just an identity + a few commands on
top of this. The base handles everything they have in common:

  * gateway wiring and intents
  * responding to @mentions, pings of any role the bot holds, and DMs
  * stripping mention tokens from the message
  * short-term per-channel/DM conversation memory
  * RAG retrieval from the agent's own Qdrant collection (with citations)
  * the Ollama chat call (system prompt + context + history, keep_alive)
  * command dispatch (``!name args``) and an automatic ``!help``
  * chunked replies (Discord's 2000-char cap)
  * blocking skill calls run off the event loop

Usage:

    from skills.discord_agent import DiscordAgent

    scout = DiscordAgent(
        name="Scout",
        system_prompt=SCOUT_SYSTEM,
        memory_collection="scout_memory",
        help_text=HELP_TEXT,
    )

    @scout.command("pipeline")
    async def pipeline(agent, args, message):
        return "..."          # the returned string is sent (chunked) as the reply

    @scout.on_startup
    async def warm(agent):
        ...                   # optional background task after the bot connects

    scout.run(token)
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict, deque
from typing import Awaitable, Callable

import aiohttp
import discord

from skills import vector_store as vs
from skills.config import config
from skills.logging import get_logger, set_agent

# Matches Discord user (<@123>, <@!123>) and role (<@&123>) mention tokens.
_MENTION_RE = re.compile(r"<@[!&]?\d+>")

CommandHandler = Callable[["DiscordAgent", str, discord.Message], Awaitable["str | None"]]
StartupHook = Callable[["DiscordAgent"], Awaitable[None]]


class DiscordAgent:
    """A conversational Discord agent backed by RAG + Ollama.

    Subclass-free: construct one, register commands with ``@agent.command(...)``,
    optionally an ``@agent.on_startup`` hook, then call ``agent.run(token)``.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        memory_collection: str,
        help_text: str = "",
        model: str | None = None,
        keep_alive: str = "30m",
        history_messages: int = 6,
        context_k: int = 4,
        min_score: float = 0.4,
        auto_channel_ids: list[int] | None = None,
    ) -> None:
        self.name = name
        # Channels where the agent answers EVERY (human) message, no @mention
        # needed -- a dedicated ChatGPT-style room.
        self.auto_channel_ids = set(auto_channel_ids or [])
        self.system_prompt = system_prompt
        self.memory_collection = memory_collection
        self.help_text = help_text or f"**{name}** -- @mention or DM me a question."
        self.model = model or config.default_model
        self.keep_alive = keep_alive
        self.context_k = context_k
        self.min_score = min_score

        set_agent(name)
        self.log = get_logger(name.lower())
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=history_messages))
        self._commands: dict[str, CommandHandler] = {}
        self._startup: StartupHook | None = None

        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)
        self.client.event(self.on_ready)
        self.client.event(self.on_message)

    # -- registration ------------------------------------------------------

    def command(self, name: str) -> Callable[[CommandHandler], CommandHandler]:
        """Register a ``!name`` command. Handler: (agent, args, message) -> reply."""
        def decorator(fn: CommandHandler) -> CommandHandler:
            self._commands[name.lower()] = fn
            return fn
        return decorator

    def on_startup(self, fn: StartupHook) -> StartupHook:
        """Register a coroutine run (in the background) once the bot connects."""
        self._startup = fn
        return fn

    # -- core capabilities (usable from command handlers) ------------------

    async def retrieve_context(self, question: str) -> tuple[str, list[str]]:
        """RAG: return (context_block, cited_titles) from this agent's collection."""
        res = await asyncio.to_thread(vs.query, self.memory_collection, question, self.context_k)
        if not res.ok:
            self.log.warning("RAG query failed: %s", res.error)
            return "", []
        hits = [h for h in res.data if (h.get("score") or 0) >= self.min_score]
        blocks, titles = [], []
        for h in hits:
            title = h["payload"].get("title") or h["doc_id"]
            text = (h["payload"].get("text") or "")[:600]
            blocks.append(f"### {title}\n{text}")
            titles.append(title)
        return "\n\n".join(blocks), titles

    async def ask(self, question: str, context: str = "", history: list[dict] | None = None) -> str:
        """Ask the local model with optional retrieved context and chat history."""
        system = self.system_prompt
        if context:
            system += (
                "\n\n## Relevant notes from Pipe's vault\n" + context +
                "\n\nUse these when relevant and name the note you drew from. If "
                "they don't cover the question, rely on what you know and say so."
            )
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{config.ollama_host}/api/chat", json=payload) as resp:
                data = await resp.json()
                return data["message"]["content"]

    # -- Discord event handlers --------------------------------------------

    async def on_ready(self) -> None:
        self.log.info("%s online as %s (id=%s)", self.name, self.client.user, self.client.user.id)
        if self._startup is not None:
            asyncio.create_task(self._run_startup())

    async def _run_startup(self) -> None:
        try:
            await self._startup(self)  # type: ignore[misc]
        except Exception:
            self.log.exception("startup hook failed")

    def _addressed(self, message: discord.Message) -> bool:
        """True if this message is for us: DM, @mention, role ping, or any
        message in a dedicated auto-respond channel."""
        if isinstance(message.channel, discord.DMChannel):
            return True
        # In a dedicated channel, answer every human message (skip other bots).
        if message.channel.id in self.auto_channel_ids and not message.author.bot:
            return True
        if self.client.user.mentioned_in(message):
            return True
        return (
            message.guild is not None
            and message.guild.me is not None
            and any(r in message.guild.me.roles for r in message.role_mentions)
        )

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.client.user:
            return
        if not self._addressed(message):
            return

        content = _MENTION_RE.sub("", message.content).strip()
        self.log.info(
            "handling from=%s channel=%s: %r",
            message.author, getattr(message.channel, "name", "dm"), content[:80],
        )
        if not content:
            await message.reply(self.help_text)
            return

        conv = self._history[message.channel.id]
        async with message.channel.typing():
            try:
                reply = await self._dispatch(content, conv, message)
            except Exception as exc:  # never let one bad message kill the bot
                self.log.exception("on_message failed")
                reply = f"Something broke handling that: {exc}"
        if reply:
            await self._reply_chunked(message, reply)

    async def _dispatch(self, content: str, conv: deque, message: discord.Message) -> str | None:
        """Route to a command, or the conversational RAG path."""
        if content.startswith("!"):
            cmd, _, args = content[1:].partition(" ")
            cmd = cmd.lower()
            if cmd == "help":
                return self.help_text
            handler = self._commands.get(cmd)
            if handler is None:
                return f"Unknown command `!{cmd}`. Try `!help`."
            return await handler(self, args.strip(), message)

        # Conversational RAG path.
        context, titles = await self.retrieve_context(content)
        try:
            answer = await self.ask(content, context, list(conv))
        except aiohttp.ClientError as exc:
            self.log.warning("ollama unreachable: %s", exc)
            return "I can't reach the model right now -- is Ollama running? Try again in a moment."
        conv.append({"role": "user", "content": content})
        conv.append({"role": "assistant", "content": answer})
        if titles:
            answer += "\n\n—\n*Drew from: " + ", ".join(titles) + "*"
        return answer

    async def post(self, channel_id: int, text: str) -> bool:
        """Proactively post to a channel by ID -- for scheduled/unprompted output.

        Returns False if the channel isn't found or visible to the bot.
        """
        channel = self.client.get_channel(channel_id)
        if channel is None:
            self.log.warning("post: channel %s not found or not visible", channel_id)
            return False
        for i in range(0, len(text), 1900):
            await channel.send(text[i:i + 1900])
        return True

    async def _reply_chunked(self, message: discord.Message, text: str) -> None:
        """Split replies over Discord's 2000-char cap."""
        if len(text) <= 1900:
            await message.reply(text)
            return
        chunks = [text[i:i + 1900] for i in range(0, len(text), 1900)]
        await message.reply(chunks[0])
        for chunk in chunks[1:]:
            await message.channel.send(chunk)

    # -- entry point -------------------------------------------------------

    def run(self, token: str | None) -> None:
        if not token:
            raise SystemExit(f"No Discord token for {self.name} (set DISCORD_TOKEN in its .env)")
        self.client.run(token)
