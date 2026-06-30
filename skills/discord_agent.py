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
from skills import obsidian_vault as vault
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
        rag_exclude_folders: list[str] | None = None,
        rag_include_folders: list[str] | None = None,
    ) -> None:
        self.name = name
        # Channels where the agent answers EVERY (human) message, no @mention
        # needed -- a dedicated ChatGPT-style room.
        self.auto_channel_ids = set(auto_channel_ids or [])
        self.system_prompt = system_prompt
        self.memory_collection = memory_collection
        # Each agent's own growing knowledge: facts it's taught + things it learns.
        # RAG'd on every chat, so the agent gets tailored to how it's actually used.
        self.learned_collection = f"{name.lower()}_memory"
        # Where this agent writes notes in the vault (one tidy place per agent),
        # so anything you ask it to save becomes real markdown you can open.
        self.notes_folder = f"Agent Notes/{name}"
        self.help_text = help_text or f"**{name}** -- @mention or DM me a question."
        self.model = model or config.default_model
        self.keep_alive = keep_alive
        self.context_k = context_k
        self.min_score = min_score
        # Vault-folder scoping for RAG, so e.g. Mason can't lead a printing answer
        # with a Spanish-class film note. include wins if set (allowlist: keep ONLY
        # these folders); otherwise exclude is a blocklist. Paths are doc_id prefixes.
        self.rag_exclude_folders = tuple(rag_exclude_folders or ())
        self.rag_include_folders = tuple(rag_include_folders or ())

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

    def _note_allowed(self, doc_id: str) -> bool:
        """Whether a note's folder is in scope for this agent's RAG. An allowlist
        (rag_include_folders) wins if set; else a blocklist (rag_exclude_folders).
        Non-vault doc_ids (e.g. learned 'taught-...') are always allowed."""
        if self.rag_include_folders:
            return doc_id.startswith(self.rag_include_folders) or not doc_id.endswith(".md")
        if self.rag_exclude_folders:
            return not doc_id.startswith(self.rag_exclude_folders)
        return True

    async def retrieve_context(self, question: str) -> tuple[str, list[str]]:
        """RAG from the agent's normal collection AND its own learned knowledge.

        Over-fetches, drops notes in excluded folders (e.g. School/ for project
        agents), then keeps the top ``context_k`` -- so filtering can't leave the
        agent staring at off-domain notes just because they scored a hair higher.
        """
        cols = [self.memory_collection]
        if self.learned_collection and self.learned_collection != self.memory_collection:
            cols.append(self.learned_collection)
        fetch = max(self.context_k * 4, 16)  # over-fetch to survive folder filtering
        blocks, titles = [], []
        for col in cols:
            try:
                res = await asyncio.to_thread(vs.query, col, question, fetch)
            except Exception:
                continue
            if not res.ok:
                self.log.warning("RAG query failed (%s): %s", col, res.error)
                continue
            kept = 0
            for h in res.data:
                if (h.get("score") or 0) < self.min_score:
                    continue
                if not self._note_allowed(h["doc_id"]):
                    continue
                title = h["payload"].get("title") or h["doc_id"]
                text = (h["payload"].get("text") or "")[:600]
                blocks.append(f"### {title}\n{text}")
                titles.append(title)
                kept += 1
                if kept >= self.context_k:
                    break
        return "\n\n".join(blocks), titles

    async def ask(self, question: str, context: str = "", history: list[dict] | None = None) -> str:
        """Ask the local model with optional retrieved context and chat history."""
        # Always ground the model in the real current date, so it doesn't treat
        # months-old notes as upcoming ("Copa 506 coming up" written in March).
        system = f"Today's date is {date.today():%A, %Y-%m-%d}.\n\n" + self.system_prompt
        system += (
            "\n\n## Honesty rules (do not break these)\n"
            "1. Refer to a note or file ONLY by the exact title/path shown in the context "
            "below. Never invent, guess, or reword a note or file name — if it isn't listed, "
            "you don't have it.\n"
            "2. Never state a specific live reading about Pipe's hardware (printer temps, print "
            "progress, server stats, status) unless it appears in the context below. If you "
            "don't have the real number, say so and point to the command that shows it.\n"
            "3. You act ONLY through `!` commands. In plain chat you cannot save notes, change "
            "settings, or run anything — so never claim you did. If asked to save something, "
            "tell Pipe the exact `!` command.\n"
            "4. If the context doesn't answer the question, say it isn't in the vault. You may "
            "then use general knowledge, but never dress it up as a note or a real reading."
        )
        # Ground the model in the commands that ACTUALLY exist, so it stops inventing
        # commands (e.g. telling Pipe to run a '!setspeed' that doesn't exist).
        cmds = ["help", "note", "learn", "recall", "tell"] + sorted(self._commands)
        system += (
            "\n\n## Your ONLY real commands\n"
            "Exactly these exist: " + "  ".join(f"`!{c}`" for c in cmds) + "\n"
            "NEVER invent a command or make up its name/syntax. If Pipe wants something none of "
            "these do, say so and point him to the closest real one (or `!help`) — do not "
            "fabricate a command."
        )
        if context:
            system += (
                "\n\n## Context retrieved for this question (the only real notes/data you have)\n"
                + context +
                "\n\nName the exact note or value you used."
            )
        else:
            system += "\n\n## Context\nNothing relevant was retrieved for this question."
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            # Match Ollama's threads to the container's cpuset, or CFS throttling
            # cripples inference (~16x slower). See config.ollama_num_threads.
            "options": {"num_thread": config.ollama_num_threads},
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

    async def _write_vault_note(self, text: str, kind: str) -> str | None:
        """Write a real markdown note to the vault (synced to the laptop), embed it
        so it's searchable, and tell Axiom. Returns the path, or None on failure.

        This is how conversations turn into real, current data Pipe can actually
        open -- not just embeddings only the agents can see."""
        from datetime import datetime as _dt
        stamp = _dt.now().strftime("%Y-%m-%d %H%M%S")
        path = f"{self.notes_folder}/{stamp}.md"
        w = await asyncio.to_thread(
            vault.write_note, path, text,
            {"type": f"{self.name.lower()}-note", "kind": kind, "tags": [self.name.lower()]})
        if not w.ok:
            self.log.warning("vault note write failed: %s", w.error)
            return None
        try:  # embed into the shared vault index so it's immediately findable
            await asyncio.to_thread(
                vs.upsert, "vault_index", path, text,
                {"title": f"{self.name} note {stamp}", "folder": self.notes_folder})
        except Exception:
            self.log.exception("vault note embed failed")
        try:  # let the librarian fold it into the Agent Notes MOC
            await asyncio.to_thread(agent_mail.send, "Axiom", self.name, "new vault note saved", path)
        except Exception:
            pass
        return path

    async def _cmd_note(self, text: str) -> str:
        """Built-in: ``!note <text>`` -- save a note to the Obsidian vault."""
        if not text:
            return "Usage: `!note <text>` — I'll save it to your Obsidian vault."
        path = await self._write_vault_note(text, "note")
        return (f"📝 Saved to your vault → `{path}` (syncs to your laptop; Axiom notified)."
                if path else "Couldn't save the note to the vault — check the logs.")

    async def _cmd_learn(self, fact: str) -> str:
        """Built-in: ``!learn <fact>`` -- remember a fact AND save it to the vault."""
        if not fact:
            return "Usage: `!learn <fact>` — tell me something to remember and I'll keep it."
        import uuid
        res = await asyncio.to_thread(
            vs.upsert, self.learned_collection, f"taught-{uuid.uuid4().hex[:8]}", fact,
            {"title": fact[:60], "kind": "taught"})
        if not res.ok:
            return f"Couldn't store that: {res.error}"
        path = await self._write_vault_note(fact, "taught")
        where = f" and saved it to your vault → `{path}`" if path else ""
        return f"🧠 Got it — I'll remember that and use it{where}."

    async def _cmd_recall(self, topic: str) -> str:
        """Built-in: ``!recall [topic]`` -- search/recall what this agent has learned."""
        if topic:
            res = await asyncio.to_thread(vs.query, self.learned_collection, topic, 5)
            hits = [h for h in (res.data or []) if (h.get("score") or 0) >= 0.3]
            if not hits:
                return f"I haven't learned anything about '{topic}' yet — teach me with `!learn`."
            return f"🧠 **What I've learned about '{topic}':**\n" + "\n".join(
                f"- {h['payload'].get('text', '')[:160]}" for h in hits)
        res = await asyncio.to_thread(vs.count, self.learned_collection)
        return (f"🧠 I've learned **{res.data if res.ok else 0}** things so far. "
                "`!recall <topic>` to search · `!learn <fact>` to teach me.")

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

    # Built-in command names (every agent has these on top of its own).
    _BUILTIN_CMDS = {"help", "tell", "learn", "recall", "knowledge",
                     "note", "newnote", "savenote"}
    # Connectives to drop if they land as a chained command's args ("!snap and !status").
    _CHAIN_FILLER = {"and", "then", "also", "&", "+", "plus", "->"}

    def _is_known_command(self, cmd: str) -> bool:
        return cmd in self._BUILTIN_CMDS or cmd in self._commands

    def _split_commands(self, text: str) -> list[tuple[str, str]]:
        """Split a message into chained commands: '!a x !b y' -> [('a','x'),('b','y')].
        Each command's args run until the next '!command'."""
        segs = []
        for m in re.finditer(r"!([A-Za-z][\w-]*)([^!]*)", text):
            args = m.group(2).strip()
            if args.lower() in self._CHAIN_FILLER:
                args = ""
            segs.append((m.group(1).lower(), args))
        return segs

    async def _save_note(self, body: str, message) -> str:
        """Save a note via the agent's own !note command if it has one, else base."""
        handler = self._commands.get("note")
        if handler is not None:
            return await handler(self, body, message)
        return await self._cmd_note(body)

    async def _run_command(self, cmd: str, args: str, message) -> str | None:
        """Execute one '!command' -- built-ins first, then the agent's own."""
        if cmd == "help":
            return self.help_text + (
                "\n**Memory & notes (I grow with use):**\n"
                "- `!note <text>` -- save a note to your Obsidian vault\n"
                "- ...or just write the whole note and end the message with `!newnote`\n"
                "- `!learn <fact>` -- teach me; I keep it, use it, and save it to your vault\n"
                "- `!recall [topic]` -- recall what I've learned\n"
                "_Tip: chain commands in one message, e.g. `!status !snap`._")
        if cmd == "tell":
            return await self._cmd_tell(args)
        if cmd == "learn":
            return await self._cmd_learn(args)
        if cmd in ("recall", "knowledge"):
            return await self._cmd_recall(args)
        if cmd in ("note", "newnote", "savenote"):
            return await self._save_note(args, message)
        handler = self._commands.get(cmd)
        if handler is None:
            return f"Unknown command `!{cmd}`. Try `!help`."
        return await handler(self, args, message)

    async def _dispatch(self, content: str, conv: deque, message: discord.Message) -> str | None:
        """Route to command(s), a trailing-note, or the conversational RAG path."""
        stripped = content.strip()

        # Trailing note: write the whole note, then end the message with !newnote.
        if not stripped.startswith("!"):
            m = re.match(r"^(.*\S)\s*!(?:newnote|savenote)\s*$",
                         stripped, re.IGNORECASE | re.DOTALL)
            if m:
                return await self._save_note(m.group(1).strip(), message)

        # One or more chained !commands.
        if stripped.startswith("!"):
            segs = self._split_commands(stripped)
            # Only treat as a chain if EVERY part is a real command (so note text
            # that happens to contain '!' isn't mistaken for a second command).
            if len(segs) >= 2 and all(self._is_known_command(c) for c, _ in segs):
                replies = []
                for c, a in segs:
                    try:
                        r = await self._run_command(c, a, message)
                    except Exception as exc:
                        self.log.exception("chained command !%s failed", c)
                        r = f"`!{c}` broke: {exc}"
                    if r:
                        replies.append(r)
                return "\n\n".join(replies) if replies else None
            cmd, _, args = stripped[1:].partition(" ")
            return await self._run_command(cmd.lower(), args.strip(), message)

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
