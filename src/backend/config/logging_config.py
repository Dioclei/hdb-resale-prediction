import datetime as dt
import json
import logging
from pathlib import Path
from typing import override

# src/backend/config/logging_config.py -> project-root (adjust .parent count if you move this file)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Attributes that exist on every LogRecord by default (i.e. NOT passed via `extra`).
# Used to figure out which attributes on the record are "extra" / custom fields.
LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}

class JSONLogFormatter(logging.Formatter):
    def __init__(self, *, fmt_keys: dict[str, str] | None = None):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.Record) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.Record) -> dict[str, str]:
       # Fields that are always present, regardless of fmt_keys config
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }
 
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)
 
        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)
 
        # fmt_keys maps output_key -> LogRecord attribute name, e.g. {"level": "levelname"}
        message_dict = {
            key: getattr(record, val)
            for key, val in self.fmt_keys.items()
            if key not in always_fields
        }
        message_dict.update(always_fields)

        # Include any custom attributes passed via `extra={...}` in the logging call
        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS and key not in message_dict:
                message_dict[key] = val

        return message_dict

class NonErrorFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord):
        return record.levelno <= logging.INFO

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "non_error": {
            "()": NonErrorFilter
        }
    },
    "formatters": {
        "simple": {
            "format": "%(asctime)s %(name)s %(levelname)s : %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
        "json": {
            "()": JSONLogFormatter,
            "fmt_keys": {
                "level": "levelname",
                "message": "message",
                "timestamp": "timestamp",
                "logger": "name",
            }
        }
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "filters": ["non_error"],
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "stderr": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "simple",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": str(LOG_DIR / "app.jsonl"), # uses absolute path
            "maxBytes": 10000,
            "backupCount": 3,
        },
        "queue_handler": {
            "class": "logging.handlers.QueueHandler",
            "handlers": [
                "stdout",
                "stderr",
                "file"
            ],
            "respect_handler_level": True,
        }
    },
    "loggers": {
        "root": {"level": "DEBUG", "handlers": ["queue_handler"]},
    },
}