# A list of test cases to evaluate the agent
GOLDEN_DATASET = [
    {
        "id": "test_01",
        "input": "What is the price of a laptop?",
        "expected_tool": "get_product_price",
        "expected_keywords": ["999", "$999"],
        "description": "Should use price tool and return correct price"
    },
    {
        "id": "test_02",
        "input": "Where is order A100?",
        "expected_tool": "check_order_status",
        "expected_keywords": ["shipped", "A100"],
        "description": "Should use order tool and return correct status"
    },
    {
        "id": "test_03",
        "input": "What is the return policy?",
        "expected_tool": "search_knowledge_base",
        "expected_keywords": ["30 days", "refund"],
        "description": "Should use RAG tool and return policy details"
    },
    {
        "id": "test_04",
        "input": "Tell me a joke about Python",
        "expected_tool": None, # Should NOT use a tool
        "expected_keywords": ["joke", "python"],
        "description": "Should use LLM directly, no tools"
    },
    {
        "id": "test_05",
        "input": "What is the price of a spaceship?",
        "expected_tool": "get_product_price",
        "expected_keywords": ["not found", "error", "unavailable"],
        "description": "Should use tool but handle unknown product gracefully"
    }
]