from retry import RetryError, retry_with_backoff
from context import RequestContext 
import logging

from config import Config
from storage import MemoryStorage
from tools import plan_tool_call, execute_tool, validate_tool_call
from llm import create_llm_client, LLMError
from prompts import build_system_prompt, build_user_prompt
from tool_schemas import TOOL_SCHEMAS
from logger import log_user_input, log_tool_planned, log_tool_result, log_llm_call, log_usage_check, log_response, log_fallback
from fallback import get_fallback_response

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
            message = get_fallback_response(
                "tool_failed",
                details=tool_result.get("message")
            )
            log_fallback("tool_failed", message)
            return message

        result = tool_result["result"]

        if tool_call["tool"] == "get_product_price":
            return f"The price of {result['product']} is ${result['price']}."

        if tool_call["tool"] == "check_order_status":
            return f"Order {result['order_id']} is {result['order_status']}."

        if tool_call["tool"] == "get_weather":
            return f"The weather in {result['city']} is {result['weather']}."

        return f"Tool result: {result}"

    def get_usage_limit(self):
        plan = self.metadata.get("user_plan", "free")
        return None if plan == "premium" else Config.MESSAGE_LIMIT_FREE

    def increment_message_used(self):
        self.metadata["messages_used"] = self.metadata.get("messages_used", 0) + 1
        self.storage.save_metadata(self.metadata)

    def check_usage_limit(self):
        limit = self.get_usage_limit()
        plan = self.metadata.get("user_plan", "free")

        if limit is not None and self.metadata.get("messages_used", 0) >= limit:
            logger.warning("Usage limit reached: %s (%s plan)", limit, plan)
            return get_fallback_response(
                "usage_limit",
                details=f"{limit} msgs for {plan} plan"
            )

        return None

    def generate_llm_response(self, prompt):
        return retry_with_backoff(
            lambda: self.llm.generate(prompt),
            max_attempts=Config.RETRY_ATTEMPTS,
            delay_seconds=Config.RETRY_DELAY,
            exceptions=LLMError,
        )

    def respond(self, user_input: str, session_id: str = None) -> str:
        ctx = RequestContext.new(session_id)

        log_user_input(user_input, ctx)

        limit_error = self.check_usage_limit()

        if limit_error:
            log_usage_check(
                plan=self.metadata.get("user_plan", "free"),
                messages_used=self.metadata.get("messages_used", 0),
                limit=self.get_usage_limit(),
                blocked=True,
                context=ctx
            )

            response = get_fallback_response("usage_limit")
            log_fallback("usage_limit", context=ctx)

            self.add_message("assistant", response)
            return response

        log_usage_check(
            plan=self.metadata.get("user_plan", "free"),
            messages_used=self.metadata.get("messages_used", 0),
            limit=self.get_usage_limit(),
            blocked=False,
            context=ctx
        )

        self.increment_message_used()
        self.add_message("user", user_input)

        tool_call = plan_tool_call(user_input)

        if tool_call:
            if not validate_tool_call(tool_call):
                response = get_fallback_response("invalid_tool")
                log_fallback("invalid_tool", details=str(tool_call), context=ctx)

                self.add_message("assistant", response)
                return response

            log_tool_planned(tool_call, ctx)

            tool_result = execute_tool(tool_call)
            success = tool_result.get("status") == "success"

            log_tool_result(
                tool_name=tool_call.get("tool"),
                success=success,
                result=tool_result,
                context=ctx
            )

            if success:
                response = self.create_tool_response(tool_call, tool_result)
            else:
                response = get_fallback_response(
                    "tool_failed",
                    details=tool_result.get("message")
                )
                log_fallback("tool_failed", details=tool_result.get("message"), context=ctx)

        else:
            try:
                prompt = self.build_prompt(user_input)
                llm_output = self.generate_llm_response(prompt)

                log_llm_call(self.llm.model_name, success=True, context=ctx)
                response = llm_output["reply"]

            except RetryError as error:
                log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
                response = get_fallback_response("llm_unavailable", details=str(error))
                log_fallback("llm_unavailable", details=str(error), context=ctx)

            except LLMError as error:
                log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
                response = get_fallback_response("llm_unavailable", details=str(error))
                log_fallback("llm_unavailable", details=str(error), context=ctx)

        self.add_message("assistant", response)
        log_response(response, ctx)

        return response