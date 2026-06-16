"""Central configuration for the homelab agent skills library.

Every skill reads its hosts, paths, and secrets from here -- never from
``os.environ`` directly. Secrets live in a ``.env`` file (gitignored); this
module loads that file once and exposes typed access to it.

Two access patterns:

    config.qdrant_host      -> optional: returns a sensible default if unset
    config.pushover_token   -> required: raises ConfigError if unset

The split is deliberate. Validation is *lazy and per-skill*: a value is only
demanded the moment a skill actually reads it. Mason never touches the Hevy
key, so Mason never fails because it's missing -- but the instant Apex tries
to read it without it being set, you get a clear error naming the missing key
and the file to put it in. No silent ``None`` surfacing as a mystery bug at 3am.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# The .env lives at the skills-library root (one level up from this package).
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing or empty."""


class Config:
    """Typed, lazily-validated access to environment configuration.

    Import the shared ``config`` instance at the bottom of this module rather
    than constructing this class yourself.
    """

    # -- internal helpers --------------------------------------------------

    def _get(self, key: str, default: str | None = None) -> str | None:
        """Return an optional value, falling back to ``default`` if unset."""
        value = os.getenv(key)
        return value if value not in (None, "") else default

    def _require(self, key: str) -> str:
        """Return a required value, or fail loudly if it's missing."""
        value = os.getenv(key)
        if value in (None, ""):
            raise ConfigError(
                f"Missing required config '{key}'. "
                f"Add it to {_ENV_PATH} (see .env.example for the template)."
            )
        return value

    # -- Logging -----------------------------------------------------------

    @property
    def agent_name(self) -> str:
        """Which agent this process is (Forge, Mason, ...). Tags every log line.

        Defaults to 'system' for standalone skill runs / tests. Each agent sets
        AGENT_NAME in its environment, or calls skills.logging.set_agent() at
        startup.
        """
        return self._get("AGENT_NAME", "system")  # type: ignore[return-value]

    @property
    def log_level(self) -> str:
        """Log level name (DEBUG/INFO/WARNING/ERROR). Defaults to INFO."""
        return self._get("LOG_LEVEL", "INFO").upper()  # type: ignore[union-attr]

    # -- Qdrant: shared vector memory --------------------------------------

    @property
    def qdrant_host(self) -> str:
        return self._get("QDRANT_HOST", "localhost")  # type: ignore[return-value]

    @property
    def qdrant_port(self) -> int:
        return int(self._get("QDRANT_PORT", "6333"))  # type: ignore[arg-type]

    @property
    def qdrant_api_key(self) -> str | None:
        return self._get("QDRANT_API_KEY")  # optional for a local instance

    # -- Ollama: shared embedding/LLM service ------------------------------

    @property
    def ollama_host(self) -> str:
        return self._get("OLLAMA_HOST", "http://localhost:11434")  # type: ignore[return-value]

    @property
    def embed_model(self) -> str:
        """Ollama model used to turn text into vectors. Default 768-dim."""
        return self._get("EMBED_MODEL", "nomic-embed-text")  # type: ignore[return-value]

    # -- Obsidian vault: shared brain --------------------------------------

    @property
    def vault_path(self) -> Path:
        return Path(self._require("OBSIDIAN_VAULT_PATH"))

    # -- Pushover: critical alerts that bypass DND -------------------------

    @property
    def pushover_token(self) -> str:
        return self._require("PUSHOVER_TOKEN")

    @property
    def pushover_user(self) -> str:
        return self._require("PUSHOVER_USER")

    # -- Discord: one webhook per channel ----------------------------------

    def discord_webhook(self, channel: str) -> str:
        """Return the webhook URL for a logical Discord channel name.

        ``"engineering"`` -> ``DISCORD_WEBHOOK_ENGINEERING``,
        ``"warden-alerts"`` -> ``DISCORD_WEBHOOK_WARDEN_ALERTS``.
        Required: raises if that channel's webhook isn't configured.
        """
        key = f"DISCORD_WEBHOOK_{channel.upper().replace('-', '_')}"
        return self._require(key)


# The single shared instance. Import THIS, not the class:
#     from skills.config import config
config = Config()
