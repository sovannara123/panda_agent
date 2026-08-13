PRODUCTS = {
    "laptop": 999,
    "phone": 699,
    "headphones": 99
}

ORDERS = {
    "A100": "shipped",
    "A101": "processing",
    "A102": "delivered"
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
    "check_order_status": check_order_status
}
def execute_tool(tool_call):
    if not isinstance(tool_call, dict):
        return {
            "status": "error",
            "message": "Tool call must be a dictionary."
        }

    tool_name = tool_call.get("tool")
    arguments = tool_call.get("arguments", {})

    if tool_name not in tool_registry:
        return {
            "status": "error",
            "message": f"Unknown tool: {tool_name}"
        }

    try:
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

    except TypeError as exc:
        return {
            "status": "error",
            "message": f"Invalid tool arguments: {exc}"
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Tool failed: {exc}"
        }