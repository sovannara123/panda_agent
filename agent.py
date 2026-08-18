import logging

from config import Config
from storage import MemoryStorage
from tools import plan_tool_call, execute_tool, validate_tool_call
from llm import create_llm_client, LLMError

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self):
        self.name = Config.AGENT_NAME
        self.storage = MemoryStorage()
        self.llm = create_llm_client()
        self.memory = self.storage.load_messages()
        self.metadata = self.storage.load_metadata()

    def add_message(self, role, content):
        self.memory.append({
            "role": role,
            "content": content
        })

        self.memory = self.memory[-Config.MAX_HISTORY:]
        self.storage.save_messages(self.memory)

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

        if tool_call["tool"] == "get_weather":
            return f"The weather in {result['city']} is {result['weather']}."

        return f"Tool result: {result}"

    def check_usage_limit(self):
        plan = self.metadata.get("user_plan", "free")
        limit = None if plan == "premium" else Config.MESSAGE_LIMIT_FREE

        if limit is not None and self.metadata.get("messages_used", 0) >= limit:
            logger.warning("Usage limit reached: %s (%s plan)", limit, plan)
            return f"Usage limit reached ({limit} msgs for {plan} plan). Please upgrade."

        return None

    def respond(self, user_input):
        logger.info("User message received: %s", user_input)

        limit_error = self.check_usage_limit()
        if limit_error:
            return limit_error

        self.metadata["messages_used"] = self.metadata.get("messages_used", 0) + 1
        self.storage.save_metadata(self.metadata)

        self.add_message("user", user_input)

        tool_call = plan_tool_call(user_input)

        if tool_call:
            if not validate_tool_call(tool_call):
                response = "I understood you want to use a tool, but the arguments were invalid. Please be more specific."
                self.add_message("assistant", response)
                return response

            logger.info("Tool call planned: %s", tool_call)
            logger.info("Tool used: %s", tool_call["tool"])
            tool_result = execute_tool(tool_call)
            logger.info("Tool result: %s", tool_result)
            if tool_result.get("status") != "success":
                logger.warning("Tool failed: %s", tool_result.get("message"))
            response = self.create_tool_response(tool_call, tool_result)
        else:
            try:
                prompt = self.build_prompt(user_input)
                llm_output = self.llm.generate(prompt)
                response = llm_output["reply"]

            except LLMError:
                logger.exception("LLM generation failed")
                logger.error("LLM failed")
                response = "Sorry, I encountered an error while generating a response."

        self.add_message("assistant", response)
        logger.info("Response sent: %s", response)

        return response