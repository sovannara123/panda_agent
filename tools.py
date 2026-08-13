
PRODUCTS = {
    "laptop": 999,
    "phone": 699,
    "headphones": 99,
    "tablet": 399,
    "mouse": 25
}

ORDERS = {
    "A100": "shipped",
    "A101": "processing",
    "A102": "delivered"
}
WEATHER = {
    "bangkok": "sunny",
    "tokyo": "rainy",
    "london": "cloudy"
}
import logging

logger = logging.getLogger(__name__)


class UnknownToolError(Exception):
    def __init__(self, tool_name):
        self.tool_name = tool_name
        super().__init__(f"Unknown tool: {tool_name}")


class ToolArgumentError(Exception):
    def __init__(self, message):
        super().__init__(message)
         
def get_weather(city):
    key = city.lower()

    if key not in WEATHER:
        return {"error": f"Weather information for '{city}' not available."}

    return {
        "city": key,
        "weather": WEATHER[key]
    }


def plan_tool_call(message):
    msg = message.lower()

    if "price" in msg:
        for product in PRODUCTS:
            if product in msg:
                return {
                    "tool": "get_product_price",
                    "arguments": {
                        "product_name": product
                    }
                }

    if "order" in msg:
        for order_id in ORDERS:
            if order_id.lower() in msg:
                return {
                    "tool": "check_order_status",
                    "arguments": {
                        "order_id": order_id
                    }
                }
    if"weather" in msg:
        for city in WEATHER:
            if city in msg:
                return {
                    "tool": "get_weather",
                    "arguments": {
                        "city": city
                    }
                }

    return None
def get_product_price(product_name):
    key = product_name.lower()

    if key not in PRODUCTS:
        return {"error": f"Product '{product_name}' not found."}

    return {
        "product": key,
        "price": PRODUCTS[key]
    }


def check_order_status(order_id):
    key = order_id.upper()

    if key not in ORDERS:
        return {"error": f"Order '{order_id}' not found."}

    return {
        "order_id": key,
        "order_status": ORDERS[key]
    }


tool_registry = {
    "get_product_price": get_product_price,
    "check_order_status": check_order_status, 
    "get_weather": get_weather
}
def execute_tool(tool_call):
    if not isinstance(tool_call, dict):
        return {
            "status": "error",
            "message": "Tool call must be a dictionary."
        }

    tool_name = tool_call.get("tool")
    arguments = tool_call.get("arguments", {})

    try:
        if tool_name not in tool_registry:
            raise UnknownToolError(tool_name)

        result = tool_registry[tool_name](**arguments)

        if isinstance(result, dict) and "error" in result:
            return {
                "status": "error",
                "message": result["error"]
            }

        return {
            "status": "success",
            "result": result
        }

    except UnknownToolError as exc:
        logger.warning("Unknown tool: %s", exc.tool_name)
        return {
            "status": "error",
            "message": str(exc)
        }

    except TypeError as exc:
        logger.warning("Invalid tool arguments: %s", exc)
        return {
            "status": "error",
            "message": f"Invalid tool arguments: {exc}"
        }

    except Exception as exc:
        logger.exception("Tool %s failed", tool_name)
        return {
            "status": "error",
            "message": f"Tool failed: {exc}"
        }