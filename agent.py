from retry import RetryError, retry_with_backoff
from context import RequestContext 
import logging
import json

from config import Config
from storage import MemoryStorage
from tools import plan_tool_call, execute_tool, validate_tool_call
from llm_adapters import create_llm_client, MockLLMClient
from prompts import build_system_prompt, build_user_prompt
from tool_schemas import TOOL_SCHEMAS
from tool_formatter import OPENAI_TOOLS
from logger import log_user_input, log_tool_planned, log_tool_result, log_llm_call, log_usage_check, log_response, log_fallback
from fallback import get_fallback_response

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, storage=None, llm_client=None):
        self.name = Config.AGENT_NAME
        self.storage = storage or MemoryStorage()
        
        if llm_client:
            self.llm = llm_client
        elif Config.API_KEY or Config.GROQ_API_KEY:
            self.llm = create_llm_client()
        else:
            self.llm = MockLLMClient(Config.MODEL_NAME)
        
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
            exceptions=Exception,
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

            except Exception as error:
                log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
                response = get_fallback_response("llm_unavailable", details=str(error))
                log_fallback("llm_unavailable", details=str(error), context=ctx)

        self.add_message("assistant", response)
        log_response(response, ctx)

        return response

    def respond(self, user_input: str, session_id: str = None) -> str:
        """Main entry point - uses function calling if available."""
        if hasattr(self.llm, 'generate_with_tools') and not isinstance(self.llm, MockLLMClient):
            return self.respond_with_function_calling(user_input, session_id)
        else:
            return self._respond_legacy(user_input, session_id)

    def _respond_legacy(self, user_input: str, session_id: str = None) -> str:
        """Legacy response method with rule-based tool planning."""
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

            except Exception as error:
                log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
                response = get_fallback_response("llm_unavailable", details=str(error))
                log_fallback("llm_unavailable", details=str(error), context=ctx)

        self.add_message("assistant", response)
        log_response(response, ctx)

        return response

    def respond_with_function_calling(self, user_input: str, session_id: str = None) -> str:
        """Main response method using LLM function calling."""
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

        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        for msg in self.memory[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        try:
            llm_response = self.llm.generate_with_tools(messages, OPENAI_TOOLS)

            log_llm_call(self.llm.model_name, success=True, context=ctx)

            if llm_response.get("tool_calls"):
                response = self._handle_tool_calls(llm_response, messages, ctx)
            else:
                response = llm_response["reply"]

        except RetryError as error:
            log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
            response = get_fallback_response("llm_unavailable")
            log_fallback("llm_unavailable", details=str(error), context=ctx)

        except Exception as error:
            log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
            response = get_fallback_response("llm_unavailable", details=str(error))
            log_fallback("llm_unavailable", details=str(error), context=ctx)

        self.add_message("assistant", response)
        log_response(response, ctx)

        return response

    def _handle_tool_calls(self, llm_response: dict, messages: list, ctx) -> str:
        """Execute tool calls and get final response from LLM."""
        tool_calls = llm_response["tool_calls"]

        # Add type field required by OpenAI API
        formatted_tool_calls = []
        for tc in tool_calls:
            formatted_tool_calls.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["tool"],
                    "arguments": tc["arguments"]
                }
            })

        messages.append({
            "role": "assistant",
            "content": llm_response.get("reply") or "",
            "tool_calls": formatted_tool_calls
        })

        for tool_call in tool_calls:
            tool_name = tool_call["tool"]

            try:
                arguments = json.loads(tool_call["arguments"])
            except json.JSONDecodeError:
                arguments = {}

            log_tool_planned({"tool": tool_name, "arguments": arguments}, ctx)

            tool_result = execute_tool({
                "tool": tool_name,
                "arguments": arguments
            })

            success = tool_result.get("status") == "success"

            log_tool_result(
                tool_name=tool_name,
                success=success,
                result=tool_result,
                context=ctx
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(tool_result)
            })

        try:
            final_response = self.llm.generate_with_tools(messages, OPENAI_TOOLS)

            log_llm_call(self.llm.model_name, success=True, context=ctx)

            return final_response["reply"]

        except Exception as error:
            log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
            return get_fallback_response("tool_failed", details=str(error))

    def respond_stream(self, user_input: str, session_id: str = None):
        """Stream response - uses function calling if available."""
        if hasattr(self.llm, 'generate_with_tools_stream') and not isinstance(self.llm, MockLLMClient):
            return self.respond_with_function_calling_stream(user_input, session_id)
        else:
            return self._respond_legacy_stream(user_input, session_id)

    def _respond_legacy_stream(self, user_input: str, session_id: str = None):
        """Legacy streaming with rule-based tool planning."""
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
            yield {"type": "content", "content": response}
            yield {"type": "done", "full_content": response}
            return

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
                yield {"type": "content", "content": response}
                yield {"type": "done", "full_content": response}
                return

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

            self.add_message("assistant", response)
            log_response(response, ctx)
            yield {"type": "content", "content": response}
            yield {"type": "done", "full_content": response}
            return

        else:
            try:
                prompt = self.build_prompt(user_input)
                for chunk in self.llm.generate_stream(prompt):
                    yield chunk
                log_llm_call(self.llm.model_name, success=True, context=ctx)

            except Exception as error:
                log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
                response = get_fallback_response("llm_unavailable", details=str(error))
                log_fallback("llm_unavailable", details=str(error), context=ctx)
                yield {"type": "content", "content": response}
                yield {"type": "done", "full_content": response}
                return

    def respond_with_function_calling_stream(self, user_input: str, session_id: str = None):
        """Stream response using LLM function calling."""
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
            yield {"type": "content", "content": response}
            yield {"type": "done", "full_content": response}
            return

        log_usage_check(
            plan=self.metadata.get("user_plan", "free"),
            messages_used=self.metadata.get("messages_used", 0),
            limit=self.get_usage_limit(),
            blocked=False,
            context=ctx
        )

        self.increment_message_used()
        self.add_message("user", user_input)

        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        for msg in self.memory[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        tool_calls_to_execute = []

        try:
            for chunk in self.llm.generate_with_tools_stream(messages, OPENAI_TOOLS):
                yield chunk

                if chunk.get("type") == "tool_calls":
                    tool_calls_to_execute = chunk["tool_calls"]

            if tool_calls_to_execute:
                yield from self._handle_tool_calls_stream(tool_calls_to_execute, messages, ctx)
            else:
                # No tool calls, just add the response to memory
                final_content = ""
                # We need to collect the content from the stream
                # The final chunk has full_content
                pass

        except Exception as error:
            log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
            response = get_fallback_response("llm_unavailable", details=str(error))
            log_fallback("llm_unavailable", details=str(error), context=ctx)
            yield {"type": "content", "content": response}
            yield {"type": "done", "full_content": response}
            return

    def _handle_tool_calls_stream(self, tool_calls: list, messages: list, ctx):
        """Execute tool calls and continue streaming with tool results."""
        formatted_tool_calls = []
        for tc in tool_calls:
            formatted_tool_calls.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["tool"],
                    "arguments": tc["arguments"]
                }
            })

        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": formatted_tool_calls
        })

        for tool_call in tool_calls:
            tool_name = tool_call["tool"]

            try:
                arguments = json.loads(tool_call["arguments"])
            except json.JSONDecodeError:
                arguments = {}

            log_tool_planned({"tool": tool_name, "arguments": arguments}, ctx)

            tool_result = execute_tool({
                "tool": tool_name,
                "arguments": arguments
            })

            success = tool_result.get("status") == "success"

            log_tool_result(
                tool_name=tool_name,
                success=success,
                result=tool_result,
                context=ctx
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(tool_result)
            })

        try:
            for chunk in self.llm.generate_with_tools_stream(messages, OPENAI_TOOLS):
                yield chunk

        except Exception as error:
            log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
            response = get_fallback_response("tool_failed", details=str(error))
            yield {"type": "content", "content": response}
            yield {"type": "done", "full_content": response}

    def respond_stream_simple(self, user_input: str, session_id: str = None):
        """Yield just content strings for simple consumers.
        
        Wraps respond_stream() and yields only the content tokens,
        skipping type fields and tool_calls events.
        """
        seen_content = False
        for chunk in self.respond_stream(user_input, session_id):
            if chunk.get("type") == "content":
                seen_content = True
                yield chunk["content"]
            elif chunk.get("type") == "done" and seen_content:
                # Only break on final done chunk (after content was seen)
                break