import asyncio
import json
import uuid

from agent import Agent
from context import RequestContext
from tool_formatter import OPENAI_TOOLS
from logger import (
    log_user_input,
    log_usage_check,
    log_tool_planned,
    log_tool_result,
    log_llm_call,
    log_response,
    log_fallback
)
from fallback import get_fallback_response
from tools import execute_tool


class AsyncAgent(Agent):
    """Async version of Agent for concurrent request handling."""

    async def respond_async(self, user_input: str, session_id: str = None) -> str:
        """Async response with function calling."""

        ctx = RequestContext.new(session_id)

        log_user_input(user_input, ctx)

        # Check usage limit
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

        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        for msg in self.memory[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Call LLM with tools
        try:
            llm_response = await self.llm.generate_with_tools_async(messages, OPENAI_TOOLS)

            log_llm_call(self.llm.model_name, success=True, context=ctx)

            # Check if LLM wants to call tools
            if llm_response.get("tool_calls"):
                response = await self._handle_tool_calls_async(llm_response, messages, ctx)
            else:
                response = llm_response["reply"]

        except Exception as error:
            log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
            response = get_fallback_response("llm_unavailable", details=str(error))
            log_fallback("llm_unavailable", details=str(error), context=ctx)

        self.add_message("assistant", response)
        log_response(response, ctx)

        return response

    async def _handle_tool_calls_async(self, llm_response: dict, messages: list, ctx) -> str:
        """Async tool call handling."""

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

        # Execute tools concurrently
        tool_results = await asyncio.gather(*[
            self._execute_single_tool_async(tool_call, ctx)
            for tool_call in tool_calls
        ])

        # Add tool results to messages
        for tool_call, tool_result in zip(tool_calls, tool_results):
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(tool_result)
            })

        # Call LLM again with tool results
        try:
            final_response = await self.llm.generate_with_tools_async(messages, OPENAI_TOOLS)

            log_llm_call(self.llm.model_name, success=True, context=ctx)

            return final_response["reply"]

        except Exception as error:
            log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
            return get_fallback_response("tool_failed", details=str(error))

    async def _execute_single_tool_async(self, tool_call: dict, ctx) -> dict:
        """Execute a single tool asynchronously."""

        tool_name = tool_call["tool"]

        try:
            arguments = json.loads(tool_call["arguments"])
        except json.JSONDecodeError:
            arguments = {}

        log_tool_planned({"tool": tool_name, "arguments": arguments}, ctx)

        # Execute tool (tools are sync, but we run them in thread pool)
        loop = asyncio.get_event_loop()
        tool_result = await loop.run_in_executor(
            None,
            execute_tool,
            {"tool": tool_name, "arguments": arguments}
        )

        success = tool_result.get("status") == "success"

        log_tool_result(
            tool_name=tool_name,
            success=success,
            result=tool_result,
            context=ctx
        )

        return tool_result

    async def respond_stream_async(self, user_input: str, session_id: str = None):
        """Async streaming response."""

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

            yield response
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

        try:
            collected_content = []
            tool_calls_data = None

            async for chunk in self.llm.generate_with_tools_stream_async(messages, OPENAI_TOOLS):
                if chunk["type"] == "content":
                    collected_content.append(chunk["content"])
                    yield chunk["content"]

                elif chunk["type"] == "tool_calls":
                    tool_calls_data = chunk["tool_calls"]

            log_llm_call(self.llm.model_name, success=True, context=ctx)

            if tool_calls_data:
                # Add type field required by OpenAI API
                formatted_tool_calls = []
                for tc in tool_calls_data:
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
                    "content": "".join(collected_content),
                    "tool_calls": formatted_tool_calls
                })

                # Execute tools concurrently
                tool_results = await asyncio.gather(*[
                    self._execute_single_tool_async(tool_call, ctx)
                    for tool_call in tool_calls_data
                ])

                for tool_call, tool_result in zip(tool_calls_data, tool_results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(tool_result)
                    })

                async for chunk in self.llm.generate_with_tools_stream_async(messages, OPENAI_TOOLS):
                    if chunk["type"] == "content":
                        yield chunk["content"]

                full_response = "".join(collected_content)

            else:
                full_response = "".join(collected_content)

            self.add_message("assistant", full_response)
            log_response(full_response, ctx)

        except Exception as error:
            log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)

            fallback = get_fallback_response("llm_unavailable", details=str(error))
            log_fallback("llm_unavailable", details=str(error), context=ctx)

            self.add_message("assistant", fallback)

            yield fallback