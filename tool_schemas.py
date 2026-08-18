TOOL_SCHEMAS = {
    "get_product_price": {
        "name": "get_product_price",
        "description": "Get the price of a product by name",
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "The exact product name (e.g., 'laptop', 'phone')"
                }
            },
            "required": ["product_name"]
        }
    },
    "check_order_status": {
        "name": "check_order_status",
        "description": "Check the shipping status of an order by ID",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order ID starting with 'A' followed by 3 digits"
                }
            },
            "required": ["order_id"]
        }
    },
    "get_weather": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name (e.g., 'bangkok', 'tokyo')"
                }
            },
            "required": ["city"]
        }
    }
}