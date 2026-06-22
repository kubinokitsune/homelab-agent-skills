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
from datetime import date
from typing import Awaitable, Callable

import aiohttp
import discord

from skills import agent_mail
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
        self._mail_handler = None       # optional inter-agent message handler
        self._attachment_handler = None  # optional file-upload handler (Codex)
        self._plain_handler = None       # optional non-command-text handler (Codex)
        self._context_provider = None    # optional extra-context source (Kairos calendar)

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

    def on_attachment(self, fn):
        """Handler for messages that carry file uploads: ``(agent, message) -> None``.

        Fires before the usual text path, so an agent like Codex can ingest
        dropped files. The handler owns the whole message when attachments exist.
        """
        self._attachment_handler = fn
        return fn

    def on_plain_message(self, fn):
        """Hook for non-command conversational text: ``(agent, content, message)
        -> bool``. Return True to fully handle it (skipping the default RAG/chat
        reply) -- e.g. Codex claiming a pasted URL or long text as a source."""
        self._plain_handler = fn
        return fn

    def on_context(self, fn):
        """Register an extra-context source: ``(agent, question) -> str``. Its
        return is appended to the RAG context on the conversational path -- e.g.
        Kairos injecting the real upcoming calendar so it stops trusting dates in
        old notes."""
        self._context_provider = fn
        return fn

    def on_mail(self, fn):
        """Register a handler for inter-agent messages: ``(agent, msg) -> None``.

        ``msg`` is a dict with from/to/subject/body. Without one, received mail
        is just surfaced in the agent's channel.
        """
        self._mail_handler = fn
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
        # Always ground the model in the real current date, so it doesn't treat
        # months-old notes as upcoming ("Copa 506 coming up" written in March).
        system = f"Today's date is {date.today():%A, %Y-%m-%d}.\n\n" + self.system_prompt
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
        asyncio.create_task(self._poll_inbox())

    async def _run_startup(self) -> None:
        try:
            await self._startup(self)  # type: ignore[misc]
        except Exception:
            self.log.exception("startup hook failed")

    async def _poll_inbox(self) -> None:
        """Check this agent's mailbox every 60s and handle new messages."""
        while True:
            await asyncio.sleep(60)
            try:
                res = await asyncio.to_thread(agent_mail.inbox, self.name)
                for msg in (res.data or []):
                    await self._handle_mail(msg)
                    await asyncio.to_thread(agent_mail.archive, msg["path"])
            except Exception:
                self.log.exception("inbox poll failed")

    async def _handle_mail(self, msg: dict) -> None:
        """Custom handler if registered, else surface the message in the channel."""
        if self._mail_handler is not None:
            await self._mail_handler(self, msg)
            return
        text = f"📨 **{msg.get('from')} → {self.name}**: {msg.get('subject')}"
        if msg.get("body"):
            text += f"\n{msg['body']}"
        for cid in self.auto_channel_ids:
            await self.post(cid, text)
            return
        self.log.info("mail from %s: %s", msg.get("from"), msg.get("subject"))

    async def _cmd_tell(self, args: str) -> str:
        """Built-in: ``!tell <Agent> <message>`` sends another agent a message."""
        to, _, body = args.partition(" ")
        if not to or not body:
            return "Usage: `!tell <Agent> <message>` -- e.g. `!tell Axiom run a duplicate scan`."
        res = await asyncio.to_thread(agent_mail.send, to.capitalize(), self.name, body)
        return f"Message on its way to {to.capitalize()}." if res.ok else f"Couldn't send: {res.error}"

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

        # File uploads: let an attachment handler (Codex) own the message.
        if message.attachments and self._attachment_handler is not None:
            try:
                await self._attachment_handler(self, message)
            except Exception:
                self.log.exception("attachment handler failed")
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
            if cmd == "tell":
                return await self._cmd_tell(args.strip())
            handler = self._commands.get(cmd)
            if handler is None:
                return f"Unknown command `!{cmd}`. Try `!help`."
            return await handler(self, args.strip(), message)

        # Let a plain-message hook (Codex) claim it -- e.g. a pasted URL/source.
        if self._plain_handler is not None:
            try:
                if await self._plain_handler(self, content, message):
                    return None
            except Exception:
                self.log.exception("plain-message handler failed")

        # Conversational RAG path.
        context, titles = await self.retrieve_context(content)
        if self._context_provider is not None:
            try:
                extra = await self._context_provider(self, content)
                if extra:
                    context = (context + "\n\n" + extra) if context else extra
            except Exception:
                self.log.exception("context provider failed")
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
