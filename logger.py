import json
import logging


def setup_logging(level=logging.INFO):
    logger = logging.getLogger("agent")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False

    return logger


log = setup_logging()


def log_tool_call(tool_name, arguments, success, duration_ms=0):
    log.info(json.dumps({
        "event": "tool_call",
        "tool": tool_name,
        "args": arguments,
        "success": success,
        "duration_ms": round(duration_ms, 2)
    }))


def log_llm_call(model, success, error=None):
    log.info(json.dumps({
        "event": "llm_call",
        "model": model,
        "success": success,
        "error": error
    }))


def log_usage_check(plan, used, limit, blocked):
    log.info(json.dumps({
        "event": "usage_check",
        "plan": plan,
        "used": used,
        "limit": limit,
        "blocked": blocked
    }))