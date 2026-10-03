"""The fleet roster, shared by the agents and the hub web UI.

Each agent opens a small HTTP bridge on 127.0.0.1:<port> (see ``DiscordAgent``)
so the hub -- running in the same container -- can talk to it without Discord.
Loopback only: nothing here is reachable from the LAN.

``deployed=False`` marks agents whose code exists but that have no service yet.

``confirm`` lists commands the bridge will NOT run until the caller confirms
(``{"confirm": true}``): things that move the printer, kill a print, restart a
service, change the firewall, or rearrange the vault. Enforced in the bridge,
not just the UI, so no client can skip it. ``safe_args`` exempts harmless
variants (``!organize dry`` only previews). Discord is unaffected.
"""

from __future__ import annotations

_MID_PRINT = " If a print is running, it will be ruined."

AGENTS: dict[str, dict] = {
    "Forge":  {"port": 8701, "emoji": "🔧", "role": "Engineering mentor",   "service": "agent-forge"},
    "Mason":  {"port": 8702, "emoji": "🧱", "role": "3D printing",          "service": "agent-mason",
               "confirm": {
                   "estop":    "EMERGENCY STOP the printer? Klipper halts immediately and needs a firmware restart afterwards." + _MID_PRINT,
                   "cancel":   "Cancel the current print? It can't be resumed.",
                   "cooldown": "Turn the heaters off?" + _MID_PRINT,
                   "print":    "Start a print? Make sure the bed is clear first.",
                   "home":     "Home all axes? The head and bed will move." + _MID_PRINT,
                   "savez":    "Save the Z-offset and restart Klipper?" + _MID_PRINT,
                   "apply":    "Apply the tuning changes to the live print?",
               }},
    "Hermes": {"port": 8703, "emoji": "🖥️", "role": "Server caretaker",     "service": "agent-hermes",
               "confirm": {
                   "restart":  "Restart this service? It will be down for a few seconds.",
               }},
    "Warden": {"port": 8704, "emoji": "🛡️", "role": "Security",             "service": "agent-warden",
               "confirm": {
                   "ban":      "Ban this IP at the firewall?",
                   "unban":    "Unban this IP? The firewall will let it back in.",
               }},
    "Axiom":  {"port": 8705, "emoji": "🗂️", "role": "Vault librarian",      "service": "agent-axiom",
               "confirm": {
                   "organize": "Run a full librarian pass? It merges duplicates (archived, never deleted), links notes and rebuilds MOCs. Tip: `!organize dry` previews first.",
                   "refile":   "Move this note to a different folder?",
                   "nightly":  "Run the nightly librarian pass now?",
               },
               "safe_args": {"organize": ["dry"]}},
    "Codex":  {"port": 8706, "emoji": "📚", "role": "Sources & research",   "service": "agent-codex"},
    "Chiron": {"port": 8707, "emoji": "🎓", "role": "Tutor",                "service": "agent-chiron"},
    "Kairos": {"port": 8708, "emoji": "📅", "role": "Scheduler",            "service": "agent-kairos",
               "confirm": {
                   "unevent":  "Delete this calendar event?",
               }},
    "Iris":   {"port": 8709, "emoji": "☀️", "role": "Morning digest",       "service": "agent-iris"},
    "Scout":  {"port": 8710, "emoji": "📋", "role": "Recruiting",           "service": "agent-scout"},
    "Apex":   {"port": 8711, "emoji": "🏋️", "role": "Gym",                  "service": "agent-apex", "deployed": False},
    "Eos":    {"port": 8712, "emoji": "🌙", "role": "Recovery",             "service": "agent-eos",  "deployed": False},
}


def bridge_port(name: str) -> int | None:
    entry = AGENTS.get(name)
    return entry["port"] if entry else None


def confirm_prompt(name: str, cmd: str, args: str = "") -> str | None:
    """The confirmation text if ``!cmd args`` on agent ``name`` needs one, else None."""
    entry = AGENTS.get(name) or {}
    prompt = (entry.get("confirm") or {}).get(cmd)
    if not prompt:
        return None
    safe = (entry.get("safe_args") or {}).get(cmd, [])
    if args.strip().lower() in safe:
        return None
    return prompt
