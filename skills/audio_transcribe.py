"""Audio -> text transcription for Codex, via faster-whisper (CPU, no GPU).

faster-whisper (CTranslate2) is the fastest CPU Whisper; it bundles audio
decoding (PyAV) so mp3/m4a/ogg/wav all work with no separate ffmpeg. The model
is lazy-loaded once and cached. Model size via config.whisper_model:
  tiny  -- fastest, roughest
  base  -- good balance (default)
  small -- more accurate, slower

    tr = audio_transcribe.transcribe(mp3_bytes, "lecture.mp3")
    if tr.ok: print(tr.data["text"])

Honest note: on a CPU (no GPU) this is roughly real-time-ish with 'base' -- a
5-minute clip takes a few minutes. First run also downloads the model (~150MB).
"""

from __future__ import annotations

import os
import tempfile

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger("audio_transcribe")

_AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac", ".wma", ".webm")
_model = None  # lazy singleton -- loading is expensive, keep it warm


def is_audio(filename: str = "", content_type: str = "") -> bool:
    """Whether a dropped file looks like audio (by extension or MIME type)."""
    return (filename or "").lower().endswith(_AUDIO_EXTS) or (content_type or "").startswith("audio/")


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel  # imported lazily so the skill loads without it
        size = config.whisper_model
        _model = WhisperModel(size, device="cpu", compute_type="int8")
        log.info("loaded whisper model '%s' (cpu/int8)", size)
    return _model


@skill
def transcribe(audio_bytes: bytes, filename: str = "audio") -> Result:
    """Transcribe audio bytes to text. Returns {text, language, duration}."""
    suffix = os.path.splitext(filename)[1].lower() or ".mp3"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            tmp = f.name
        model = _get_model()
        # beam_size=1 is much faster on CPU with little accuracy loss for clear speech.
        segments, info = model.transcribe(tmp, beam_size=1, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Result.success({
            "text": text,
            "language": getattr(info, "language", "?"),
            "duration": round(getattr(info, "duration", 0) or 0, 1),
        })
    except Exception as exc:
        return Result.failure(f"transcription failed: {exc}")
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
