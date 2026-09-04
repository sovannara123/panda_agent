SYSTEM_PROMPT = """You are {agent_name}, a helpful support assistant.

AVAILABLE TOOLS:
{tool_descriptions}

RULES:
- If the user asks about product prices, use get_product_price
- If the user asks about order status, use check_order_status
- If the user asks about weather, use get_weather
- For all other questions, answer directly
- Never make up prices or order statuses
- Be concise and friendly"""

USER_PROMPT_TEMPLATE = """Conversation history:
{history}

Current user message: {user_message}     
 
Respond according to the system rules."""


def format_tool_descriptions(tool_schemas: dict) -> str:
    lines = []

    for tool_name, schema in tool_schemas.items():
        description = schema.get("description", "No description.")
        parameters = schema.get("parameters", {})
        properties = parameters.get("properties", {})

        param_names = ", ".join(properties.keys())

        lines.append(f"- {tool_name}({param_names}): {description}")

    return "\n".join(lines)

def build_system_prompt(agent_name, schemas):
    return SYSTEM_PROMPT.format(
        agent_name=agent_name,
        tool_descriptions=format_tool_descriptions(schemas)
    )


def build_user_prompt(history, user_message):
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
    return USER_PROMPT_TEMPLATE.format(
        history=history_text,
        user_message=user_message
    )