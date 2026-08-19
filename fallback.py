FALLBACK_RESPONSES = {
    "tool_failed": (
        "Sorry, that action failed. "
        "You can try again or ask in a different way."
    ),

    "llm_unavailable": (
        "Sorry, the AI service is unavailable right now. "
        "You can still ask about product price or order status."
    ),

    "invalid_tool": (
        "I could not process that tool request. "
        "Try: 'price of laptop' or 'check order A100'."
    ),

    "usage_limit": (
        "You have reached the free message limit. "
        "Upgrade to continue."
    ),

    "unknown": (
        "I am not sure how to help with that yet."
    )
}


def get_fallback_response(reason: str, details: str = None) -> str:
    base_response = FALLBACK_RESPONSES.get(
        reason,
        FALLBACK_RESPONSES["unknown"]
    )

    if details:
        return f"{base_response} Details: {details}"

    return base_response