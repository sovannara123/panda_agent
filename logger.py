import logging
import json
from context import RequestContext 

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


def log_event(event_type: str, data: dict, context: RequestContext = None):
    base_data = {
        "event": event_type,
        **data
    }

    if context:
        base_data["session_id"] = context.session_id
        base_data["request_id"] = context.request_id

    log.info(json.dumps(base_data))

def log_user_input(user_input: str ,  context:RequestContext = None ):
    log_event("user_input", {
        "message": user_input
    } , context)


def log_usage_check(plan: str, messages_used: int, limit: int, blocked: bool , context: RequestContext = None):
    log_event("usage_check", {
        "plan": plan,
        "messages_used": messages_used,
        "limit": limit,
        "blocked": blocked
    } , context)


def log_usage_blocked(plan: str, messages_used: int, limit: int , context:RequestContext = None):
    log_event("usage_blocked", {
        "plan": plan,
        "messages_used": messages_used,
        "limit": limit
    },context )


def log_tool_planned(tool_call: dict , context:RequestContext = None):
    log_event("tool_planned", {
        "tool_call": tool_call
    }, context)


def log_tool_result(tool_name: str, success: bool, result: dict , context:RequestContext = None):
    log_event("tool_result", {
        "tool": tool_name,
        "success": success,
        "result": result
    }, context)


def log_tool_failed(tool_name: str, message: str, context: RequestContext = None):
    log_event("tool_failed", {
        "tool": tool_name,
        "message": message
    }, context)


def log_tool_call(tool_name, arguments, success, duration_ms=0, context: RequestContext = None):
    log_event("tool_call", {
        "tool": tool_name,
        "args": arguments,
        "success": success,
        "duration_ms": round(duration_ms, 2)
    }, context)


def log_llm_call(model_name: str, success: bool, error: str = None, context: RequestContext = None):
    log_event("llm_call", {
        "model": model_name,
        "success": success,
        "error": error
    }, context)


def log_response(response: str, context: RequestContext = None):
    log_event("response_sent", {
        "response": response
    }, context)


def log_fallback(reason: str, message: str = None, details: str = None, context: RequestContext = None):
    log_event("fallback", {
        "reason": reason,
        "message": message,
        "details": details
    }, context) 