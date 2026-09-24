"""
config/logging_config.py
Centralized logging with ANSI-colored terminal output for VoiceAgent.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_CONFIGURED = False

# ── ANSI codes ─────────────────────────────────────────────────────────────────
_RST  = '\033[0m'
_BOLD = '\033[1m'
_DIM  = '\033[2m'

_GREY   = '\033[38;5;244m'
_WHITE  = '\033[38;5;255m'
_CYAN   = '\033[38;5;51m'
_DCYAN  = '\033[38;5;39m'
_VIOLET = '\033[38;5;141m'
_AMBER  = '\033[38;5;214m'
_RED    = '\033[38;5;196m'
_PINK   = '\033[38;5;201m'

# (color, glyph, padded-label)
_LEVELS = {
    'DEBUG':    (_DCYAN,  '◈', 'DEBUG  '),
    'INFO':     (_CYAN,   '▶', 'INFO   '),
    'WARNING':  (_AMBER,  '⚠', 'WARN   '),
    'ERROR':    (_RED,    '✖', 'ERROR  '),
    'CRITICAL': (_PINK,   '☢', 'CRIT   '),
}

_BANNER = (
    f"\n{_CYAN}"
    "  ┌──────────────────────────────────────────┐\n"
    "  │  VoiceAgent · real-time voice assistant  │\n"
    "  └──────────────────────────────────────────┘"
    f"{_RST}\n"
)


def _prepare_stderr() -> bool:
    """
    Make stderr safe for the glyphs used below and report whether ANSI colour
    should be used. On Windows a redirected stream defaults to the ANSI code
    page, which cannot encode the log glyphs — force UTF-8 instead.
    """
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    if os.getenv("NO_COLOR") or not sys.stderr.isatty():
        return False
    if sys.platform == "win32":
        os.system("")  # enables VT100 escape processing in legacy consoles
    return True


class PlainFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            fmt="%(levelname)-8s %(asctime)s | %(name)-28s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class VoiceAgentFormatter(logging.Formatter):
    """Futuristic ANSI-colored formatter for terminal output."""

    def format(self, record: logging.LogRecord) -> str:
        color, glyph, label = _LEVELS.get(record.levelname, (_WHITE, '?', record.levelname))

        ts   = self.formatTime(record, '%H:%M:%S')
        name = record.name.replace('voiceagent.', '')[:26]
        msg  = record.getMessage()

        if record.levelno >= logging.ERROR:
            msg_color = _RED
        elif record.levelno >= logging.WARNING:
            msg_color = _AMBER
        else:
            msg_color = _WHITE

        line = (
            f"{_DIM}[VoiceAgent]{_RST} {_GREY}{ts}{_RST}  "
            f"{color}{_BOLD}{glyph}{_RST}  {color}{label}{_RST}"
            f"{_DIM} │ {_RST}{_VIOLET}{name:<26}{_RST}"
            f"{_DIM} │ {_RST}{msg_color}{msg}{_RST}"
        )

        if record.exc_info:
            line += '\n' + self.formatException(record.exc_info)
        return line


def setup_logging() -> None:
    """
    Configure root logger once for the entire application.
    Terminal: rich ANSI-colored output.
    File: plain rotating log under logs/voiceagent.log.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level      = getattr(logging, log_level_name, logging.INFO)

    # ── Stream handler (stderr) — colored ─────────────────────────────────────
    use_color = _prepare_stderr()
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(VoiceAgentFormatter() if use_color else PlainFormatter())
    stream_handler.setLevel(log_level)

    # ── Rotating file handler — plain text ─────────────────────────────────────
    log_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "voiceagent.log")

    plain_fmt = PlainFormatter()
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(plain_fmt)
    file_handler.setLevel(log_level)

    # ── Root logger ────────────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "aiohttp", "asyncio",
                  "faster_whisper", "google_genai", "primp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if use_color:
        print(_BANNER, file=sys.stderr)

    logging.getLogger("voiceagent").info(
        "System online  level=%s  logfile=%s", log_level_name, log_path
    )


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'voiceagent' namespace."""
    return logging.getLogger(f"voiceagent.{name}")
