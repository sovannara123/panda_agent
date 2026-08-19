import logging
import json


def setup_logger(name="agent", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False

    return logger


log = setup_logger()


def log_event(event_type: str, data: dict):
    log.info(json.dumps({
        "event": event_type,
        **data
    }))


def log_user_input(user_input: str):
    log_event("user_input", {
        "message": user_input
    })


def log_usage_check(plan: str, messages_used: int, limit: int, blocked: bool):
    log_event("usage_check", {
        "plan": plan,
        "messages_used": messages_used,
        "limit": limit,
        "blocked": blocked
    })


def log_usage_blocked(plan: str, messages_used: int, limit: int):
    log_event("usage_blocked", {
        "plan": plan,
        "messages_used": messages_used,
        "limit": limit
    })


def log_tool_planned(tool_call: dict):
    log_event("tool_planned", {
        "tool_call": tool_call
    })


def log_tool_result(tool_name: str, success: bool, result: dict):
    log_event("tool_result", {
        "tool": tool_name,
        "success": success,
        "result": result
    })


def log_tool_failed(tool_name: str, message: str):
    log_event("tool_failed", {
        "tool": tool_name,
        "message": message
    })


def log_tool_call(tool_name, arguments, success, duration_ms=0):
    log_event("tool_call", {
        "tool": tool_name,
        "args": arguments,
        "success": success,
        "duration_ms": round(duration_ms, 2)
    })


def log_llm_call(model_name: str, success: bool, error: str = None):
    log_event("llm_call", {
        "model": model_name,
        "success": success,
        "error": error
    })


def log_response(response: str):
    log_event("response_sent", {
        "response": response
    })