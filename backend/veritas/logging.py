"""Structured logging.

Human-readable by default; set ``VERITAS_LOG_JSON=true`` for machine-parseable
output when shipping to a log aggregator.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_CONFIGURED = False

_LEVEL_COLOURS = {
    "DEBUG": "\033[38;5;244m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;196m",
    "CRITICAL": "\033[48;5;196m\033[38;5;231m",
}
_RESET = "\033[0m"
_DIM = "\033[38;5;244m"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def __init__(self, use_colour: bool) -> None:
        super().__init__()
        self.use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        name = record.name.replace("veritas.", "")
        level = record.levelname
        message = record.getMessage()

        extra = getattr(record, "extra_fields", {})
        suffix = ""
        if extra:
            suffix = " " + " ".join(f"{k}={v}" for k, v in extra.items())

        if self.use_colour:
            colour = _LEVEL_COLOURS.get(level, "")
            line = (
                f"{_DIM}{ts}{_RESET} {colour}{level:<7}{_RESET} "
                f"{_DIM}{name:<22}{_RESET} {message}{_DIM}{suffix}{_RESET}"
            )
        else:
            line = f"{ts} {level:<7} {name:<22} {message}{suffix}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: str = "INFO", as_json: bool = False) -> None:
    """Install the root handler. Idempotent — safe to call from any entrypoint."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    if as_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(ConsoleFormatter(use_colour=sys.stderr.isatty()))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Third-party libraries are chatty at INFO and drown out our own events.
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "anthropic", "trafilatura"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


class BoundLogger:
    """Thin wrapper that attaches structured fields to every record."""

    __slots__ = ("_logger", "_fields")

    def __init__(self, logger: logging.Logger, fields: dict[str, Any] | None = None) -> None:
        self._logger = logger
        self._fields = fields or {}

    def bind(self, **fields: Any) -> BoundLogger:
        return BoundLogger(self._logger, {**self._fields, **fields})

    def _log(self, level: int, message: str, **fields: Any) -> None:
        self._logger.log(
            level, message, extra={"extra_fields": {**self._fields, **fields}}, stacklevel=3
        )

    def debug(self, message: str, **fields: Any) -> None:
        self._log(logging.DEBUG, message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self._log(logging.INFO, message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._log(logging.WARNING, message, **fields)

    def error(self, message: str, exc_info: bool = False, **fields: Any) -> None:
        self._logger.error(
            message,
            extra={"extra_fields": {**self._fields, **fields}},
            exc_info=exc_info,
            stacklevel=3,
        )


def get_logger(name: str, **fields: Any) -> BoundLogger:
    return BoundLogger(logging.getLogger(name), fields)
