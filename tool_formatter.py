from tool_schemas import TOOL_SCHEMAS


def convert_to_openai_tools(schemas: dict) -> list:
    """Convert our tool schemas to OpenAI function calling format."""

    openai_tools = []

    for tool_name, schema in schemas.items():
        openai_tool = {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"]
            }
        }

        openai_tools.append(openai_tool)

    return openai_tools


OPENAI_TOOLS = convert_to_openai_tools(TOOL_SCHEMAS)