"""Logging setup with secret redaction."""
from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_SECRET_ENV = ("EARTHDATA_TOKEN", "EARTHDATA_PASSWORD")
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+")
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class RedactingFilter(logging.Filter):
    """Mask credentials that appear in log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for name in _SECRET_ENV:
            secret = os.environ.get(name)
            if secret and len(secret) >= 4:
                message = message.replace(secret, "***REDACTED***")
        message = _BEARER.sub(r"\1***REDACTED***", message)
        record.msg, record.args = message, None
        return True


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        if getattr(handler, "_terramind", False):
            root.removeHandler(handler)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"))

    for handler in handlers:
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler.addFilter(RedactingFilter())
        handler._terramind = True  # type: ignore[attr-defined]
        root.addHandler(handler)
