import logging
import time # used to measure tool execution time

from config import Config
from storage import MemoryStorage
from tools import plan_tool_call, execute_tool, validate_tool_call
from llm import create_llm_client, LLMError
from prompts import build_system_prompt, build_user_prompt
from tool_schemas import TOOL_SCHEMAS
from logger import log_tool_call, log_llm_call, log_usage_check

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, storage=None):
        self.name = Config.AGENT_NAME
        self.storage = storage or MemoryStorage()
        self.llm = create_llm_client()
        self.memory = self.storage.load_messages()
        self.metadata = self.storage.load_metadata()
        self.system_prompt = build_system_prompt(self.name, TOOL_SCHEMAS)

    def add_message(self, role, content):
        self.memory.append({
            "role": role,
            "content": content
        })

        self.memory = self.memory[-Config.MAX_HISTORY:]
        self.storage.save_messages(self.memory)

    def build_prompt(self, user_input):
        return build_user_prompt(self.memory[-5:], user_input)

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
        limit_error = self.check_usage_limit() # to check the usage limit before processing the user input
        plan = self.metadata.get("user_plan", "free") # to get the user's plan
        limit = None if plan == "premium" else Config.MESSAGE_LIMIT_FREE # to get the message limit based on the user's plan
        log_usage_check( # to log the usage check event
            plan,
            self.metadata.get("messages_used", 0),
            limit,
            limit_error is not None
        )
        if limit_error: # if the usage limit is reached, return the error message
            return limit_error

        self.metadata["messages_used"] = self.metadata.get("messages_used", 0) + 1 # to increment the message count
        self.storage.save_metadata(self.metadata)       # to save the updated metadata

        self.add_message("user", user_input)

        tool_call = plan_tool_call(user_input) # to plan the tool call based on the user input

        if tool_call:
            if not validate_tool_call(tool_call): 
                response = "I understood you want to use a tool, but the arguments were invalid. Please be more specific."
                self.add_message("assistant", response) # to add the assistant's response to the memory
                return response


            start = time.time() # to measure the execution time of the tool call
            tool_result = execute_tool(tool_call)  # to execute the tool call and get the result 
            duration = (time.time() - start) * 1000 # to calculate the duration in milliseconds
            log_tool_call( # to log the tool call event
                tool_call["tool"],
                tool_call["arguments"],
                tool_result.get("status") == "success",
                duration
            )
            if tool_result.get("status") != "success":
                logger.warning("Tool failed: %s", tool_result.get("message"))
            response = self.create_tool_response(tool_call, tool_result)
        else:
            try:
                prompt = self.build_prompt(user_input)
                llm_output = self.llm.generate(prompt)
                log_llm_call(self.llm.model_name, True)
                response = llm_output["reply"]

            except LLMError as exc:
                log_llm_call(self.llm.model_name, False, str(exc))
                logger.exception("LLM generation failed")
                response = "Sorry, I encountered an error while generating a response."

        self.add_message("assistant", response)

        return response