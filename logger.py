import logging
import json
import os
import sys
from logging.handlers import RotatingFileHandler
from context import RequestContext
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from config import Config


class JsonFormatter(logging.Formatter):
    """JSON log formatter with correlation IDs."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation IDs if present
        if hasattr(record, "session_id"):
            log_data["session_id"] = getattr(record, "session_id")
        if hasattr(record, "request_id"):
            log_data["request_id"] = getattr(record, "request_id")

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "name", "pathname", "process", "processName",
                "relativeCreated", "thread", "threadName", "exc_info",
                "exc_text", "stack_info", "session_id", "request_id"
            }:
                log_data[key] = value

        return json.dumps(log_data, ensure_ascii=False)


class CorrelationFilter(logging.Filter):
    """Filter to inject correlation IDs from context."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Default values
        record.session_id = getattr(record, "session_id", "-")
        record.request_id = getattr(record, "request_id", "-")
        return True


def _get_formatter(json_format: bool, datefmt: str = "%H:%M:%S") -> logging.Formatter:
    if json_format:
        return JsonFormatter(datefmt=datefmt)
    return logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(session_id)s | %(request_id)s | %(message)s",
        datefmt=datefmt,
    )


def setup_logger(
    name: str = "agent",
    level: int | str | None = None,
    log_file: str | None = None,
    json_format: bool = False,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    datefmt: str = "%H:%M:%S",
) -> logging.Logger:
    """Configure and return a logger with console + optional file output."""
    logger = logging.getLogger(name)

    # Determine log level
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)  # type: ignore
    logger.handlers.clear()  # allow reconfiguration

    # Add correlation filter
    logger.addFilter(CorrelationFilter())

    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_get_formatter(json_format, datefmt))
    logger.addHandler(console_handler)

    # File handler (if configured)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(_get_formatter(json_format, datefmt))
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# Module-level logger (configured lazily)
_log: logging.Logger | None = None


def get_logger(name: str = "agent") -> logging.Logger:
    """Get or create the module logger (configured on first use)."""
    global _log
    if _log is None:
        from config import get_config
        cfg = get_config()
        _log = setup_logger(
            name=name,
            level=cfg.LOG_LEVEL,
            log_file=cfg.LOG_FILE,
            json_format=cfg.LOG_JSON,
        )
    return _log


def _get_base_data(event_type: str, data: dict, context: Optional[RequestContext] = None) -> dict:
    """Build base log data with correlation IDs."""
    base_data = {
        "event": event_type,
        **data
    }
    if context:
        base_data["session_id"] = context.session_id
        base_data["request_id"] = context.request_id
    return base_data


def log_event(event_type: str, data: dict, context: Optional[RequestContext] = None):
    """Log a generic event with optional correlation context."""
    log = get_logger("agent")
    log.info(json.dumps(_get_base_data(event_type, data, context)))


def log_user_input(user_input: str, context: Optional[RequestContext] = None):
    log_event("user_input", {"message": user_input}, context)


def log_usage_check(plan: str, messages_used: int, limit: int | None, blocked: bool, context: Optional[RequestContext] = None):
    log_event("usage_check", {
        "plan": plan,
        "messages_used": messages_used,
        "limit": limit,
        "blocked": blocked
    }, context)


def log_usage_blocked(plan: str, messages_used: int, limit: int | None, context: Optional[RequestContext] = None):
    log_event("usage_blocked", {
        "plan": plan,
        "messages_used": messages_used,
        "limit": limit
    }, context)


def log_tool_planned(tool_call: dict, context: Optional[RequestContext] = None):
    log_event("tool_planned", {"tool_call": tool_call}, context)


def log_tool_result(tool_name: str, success: bool, result: dict, context: Optional[RequestContext] = None):
    log_event("tool_result", {
        "tool": tool_name,
        "success": success,
        "result": result
    }, context)


def log_tool_failed(tool_name: str, message: str, context: Optional[RequestContext] = None):
    log_event("tool_failed", {
        "tool": tool_name,
        "message": message
    }, context)


def log_tool_call(tool_name, arguments, success, duration_ms=0, context: Optional[RequestContext] = None):
    log_event("tool_call", {
        "tool": tool_name,
        "args": arguments,
        "success": success,
        "duration_ms": round(duration_ms, 2)
    }, context)


def log_llm_call(model_name: str, success: bool, error: Optional[str] = None, context: Optional[RequestContext] = None):
    log_event("llm_call", {
        "model": model_name,
        "success": success,
        "error": error
    }, context)


def log_response(response: str, context: Optional[RequestContext] = None):
    log_event("response_sent", {"response": response}, context)


def log_fallback(reason: str, message: Optional[str] = None, details: Optional[str] = None, context: Optional[RequestContext] = None):
    log_event("fallback", {
        "reason": reason,
        "message": message,
        "details": details
    }, context)


# Backward compatibility
log = get_logger("agent")