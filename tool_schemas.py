TOOL_SCHEMAS = {
        "search_knowledge_base": {
        "name": "search_knowledge_base",
        "description": "Search the internal knowledge base for company policies, return rules, or product manual details.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The specific question or topic to search for (e.g., 'return policy', 'laptop battery life')"
                }
            },
            "required": ["query"]
        }
    },
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
    },
    "test_failure": {
        "name": "test_failure",
        "description": "A test tool that always fails, used to verify graceful degradation",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}



