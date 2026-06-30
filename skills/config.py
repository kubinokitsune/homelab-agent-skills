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

    @property
    def default_model(self) -> str:
        """Default Ollama chat model for conversational agents (override per-agent)."""
        return self._get("DEFAULT_AGENT_MODEL", "llama3.1:8b")  # type: ignore[return-value]

    @property
    def ollama_num_threads(self) -> int:
        """CPU threads Ollama should use per request.

        MUST match the cores the Ollama container is pinned to. On this box the
        agents LXC is cpuset 0-2 (3 cores); leaving Ollama at its auto-detected 4
        threads oversubscribes the cpuset and triggers CFS throttle-stalls that
        cut inference ~16x (0.6 vs 9.6 tok/s). Default 3 to match.
        """
        return int(self._get("OLLAMA_NUM_THREADS", "3"))  # type: ignore[arg-type]

    # -- Moonraker / 3D printer (Mason) ------------------------------------

    @property
    def moonraker_url(self) -> str:
        """Base URL of Moonraker (Klipper API). Mason talks to the printer LXC.

        Defaults to localhost; on the server set MOONRAKER_URL to the printer
        container, e.g. http://192.168.1.136:7125.
        """
        return self._get("MOONRAKER_URL", "http://127.0.0.1:7125")  # type: ignore[return-value]

    @property
    def camera_snapshot_url(self) -> str:
        """Single-frame JPEG endpoint for first-layer vision.

        ustreamer/Crowsnest expose ``/snapshot``. Defaults to the host ustreamer.
        """
        return self._get("CAMERA_SNAPSHOT_URL", "http://192.168.1.135:8080/snapshot")  # type: ignore[return-value]

    @property
    def vision_model(self) -> str:
        """Local Ollama vision model Mason uses to look at prints. Default moondream."""
        return self._get("VISION_MODEL", "moondream")  # type: ignore[return-value]

    @property
    def failure_model_path(self) -> Path:
        """Where the trained print-failure classifier is saved."""
        raw = self._get("FAILURE_MODEL_PATH")
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parent.parent.parent / "agent-data" / "failure_detector.joblib"

    @property
    def print_anomaly_model_path(self) -> Path:
        """Where the print-quality anomaly model (trained on clean prints) is saved."""
        raw = self._get("PRINT_ANOMALY_MODEL_PATH")
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parent.parent.parent / "agent-data" / "print_anomaly.joblib"

    @property
    def train_min_good(self) -> int:
        """Labeled 'good' prints needed before the failure detector is trainable."""
        return int(self._get("TRAIN_MIN_GOOD", "10"))  # type: ignore[arg-type]

    @property
    def train_min_bad(self) -> int:
        """Labeled 'bad' (failed) prints needed -- the scarce, critical class."""
        return int(self._get("TRAIN_MIN_BAD", "5"))  # type: ignore[arg-type]

    @property
    def print_dataset_dir(self) -> Path:
        """Where Mason saves labeled print frames to train a failure detector.

        Defaults to ``agent-data/print-dataset/`` beside the skills library --
        machine output, outside the Obsidian vault.
        """
        raw = self._get("PRINT_DATASET_DIR")
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parent.parent.parent / "agent-data" / "print-dataset"

    # -- Server monitoring (Hermes) ----------------------------------------

    @property
    def server_host(self) -> str:
        """Proxmox host IP Hermes manages (SSH target for host/container ops)."""
        return self._get("SERVER_HOST", "192.168.1.135")  # type: ignore[return-value]

    @property
    def server_ssh_key(self) -> str:
        """SSH key Hermes uses to reach the Proxmox host."""
        return self._get("SERVER_SSH_KEY", "/root/.ssh/hermes_host")  # type: ignore[return-value]

    @property
    def metrics_path(self) -> Path:
        """Rolling JSONL of server metrics Hermes records for anomaly detection."""
        raw = self._get("METRICS_PATH")
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parent.parent.parent / "agent-data" / "server_metrics.jsonl"

    @property
    def anomaly_model_path(self) -> Path:
        """Where Hermes's trained Isolation-Forest anomaly model is saved."""
        raw = self._get("ANOMALY_MODEL_PATH")
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parent.parent.parent / "agent-data" / "anomaly_detector.joblib"

    # -- File ops: allowlisted server roots --------------------------------

    @property
    def file_ops_roots(self) -> tuple[Path, ...]:
        """Directories file_ops is allowed to touch (OS-pathsep separated).

        Empty by default -- file_ops refuses to operate until you set
        FILE_OPS_ROOTS, so nothing roams the filesystem unintentionally.
        """
        raw = self._get("FILE_OPS_ROOTS")
        if not raw:
            return ()
        return tuple(Path(p).resolve() for p in raw.split(os.pathsep) if p.strip())

    # -- Web search --------------------------------------------------------

    @property
    def search_backend(self) -> str:
        """Which web-search backend to use. Default 'duckduckgo' (no key)."""
        return self._get("SEARCH_BACKEND", "duckduckgo").lower()  # type: ignore[union-attr]

    @property
    def search_api_key(self) -> str | None:
        """API key for a paid backend (Brave/SerpAPI). Unused by duckduckgo."""
        return self._get("SEARCH_API_KEY")

    # -- Obsidian vault: shared brain --------------------------------------

    @property
    def vault_path(self) -> Path:
        return Path(self._require("OBSIDIAN_VAULT_PATH"))

    # -- Agent reports: daily handoff for Iris -----------------------------

    @property
    def agent_reports_dir(self) -> Path:
        """Where agents write daily reports for Iris to compile.

        Defaults to ``agent-reports/`` beside the skills library (outside the
        Obsidian vault, so it doesn't pollute Axiom's index).
        """
        raw = self._get("AGENT_REPORTS_DIR")
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parent.parent.parent / "agent-reports"

    @property
    def agent_mail_dir(self) -> Path:
        """Where agents drop messages for each other (the spider web).

        Defaults to ``agent-mail/`` beside the skills library, like agent-reports.
        """
        raw = self._get("AGENT_MAIL_DIR")
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parent.parent.parent / "agent-mail"

    @property
    def build_tracker_path(self) -> Path:
        """Forge's build step-tracking store (JSON).

        Defaults to ``agent-data/forge_builds.json`` beside the skills library.
        """
        raw = self._get("BUILD_TRACKER_PATH")
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parent.parent.parent / "agent-data" / "forge_builds.json"

    @property
    def parts_list_path(self) -> Path:
        """Forge's engineering shopping list (JSON source of truth).

        Defaults to ``agent-data/forge_parts.json`` beside the skills library.
        """
        raw = self._get("PARTS_LIST_PATH")
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parent.parent.parent / "agent-data" / "forge_parts.json"

    @property
    def flashcards_path(self) -> Path:
        """Chiron's spaced-repetition flashcard store (JSON)."""
        raw = self._get("FLASHCARDS_PATH")
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parent.parent.parent / "agent-data" / "flashcards.json"

    @property
    def calendar_path(self) -> Path:
        """The shared calendar JSON (source of truth for dates/events).

        Defaults to ``agent-data/calendar.json`` beside the skills library.
        """
        raw = self._get("CALENDAR_PATH")
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parent.parent.parent / "agent-data" / "calendar.json"

    @property
    def error_log_dir(self) -> Path:
        """Where agents log errors/confusions for pattern clustering (Forge et al.).

        Defaults to ``agent-data/errors/`` beside the skills library -- machine
        output, outside the Obsidian vault like agent-reports and agent-mail.
        """
        raw = self._get("ERROR_LOG_DIR")
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parent.parent.parent / "agent-data" / "errors"

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
