from config import Config
from storage import MemoryStorage
from tools import plan_tool_call, execute_tool
from llm import MockLLMClient, LLMError


class Agent:
    def __init__(self):
        self.name = Config.AGENT_NAME
        self.storage = MemoryStorage()
        self.llm = MockLLMClient(Config.MODEL_NAME)
        self.memory = self.storage.load()

    def add_message(self, role, content):
        self.memory.append({
            "role": role,
            "content": content
        })

        self.memory = self.memory[-Config.MAX_HISTORY:]
        self.storage.save(self.memory)

    def build_prompt(self, user_input):
        recent_messages = self.memory[-5:]

        history_text = "\n".join([
            f"{item['role']}: {item['content']}"
            for item in recent_messages
        ])

        return f"Conversation history:\n{history_text}\n\nuser: {user_input}"

    def create_tool_response(self, tool_call, tool_result):
        if tool_result.get("status") != "success":
            return f"Sorry, the tool failed: {tool_result.get('message', 'Unknown error')}"

        result = tool_result["result"]

        if tool_call["tool"] == "get_product_price":
            return f"The price of {result['product']} is ${result['price']}."

        if tool_call["tool"] == "check_order_status":
            return f"Order {result['order_id']} is {result['order_status']}."

        return f"Tool result: {result}"

    def respond(self, user_input):
        self.add_message("user", user_input)

        tool_call = plan_tool_call(user_input)

        if tool_call:
            tool_result = execute_tool(tool_call)
            response = self.create_tool_response(tool_call, tool_result)
        else:
            try:
                prompt = self.build_prompt(user_input)
                llm_output = self.llm.generate(prompt)
                response = llm_output["reply"]

            except LLMError:
                response = "Sorry, I encountered an error while generating a response."

        self.add_message("assistant", response)

        return response