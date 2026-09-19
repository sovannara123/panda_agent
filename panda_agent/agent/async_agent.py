import asyncio
import json
from panda_agent.agent.agent import Agent
from panda_agent.core.context import RequestContext
from panda_agent.tools.formatter import OPENAI_TOOLS
from panda_agent.core.logger import (
    log_user_input,
    log_usage_check,
    log_tool_planned,
    log_tool_result,
    log_llm_call,
    log_response,
    log_fallback,
    log_event
)
from panda_agent.llm.fallback import get_fallback_response
from panda_agent.tools.tools import execute_tool
from panda_agent.core.config import get_config
from panda_agent.core.observability import tracer, LLM_LATENCY, TOKEN_USAGE, ACTIVE_SESSIONS
import time

class AsyncAgent(Agent):
    """Async version of Agent for concurrent request handling."""

    async def respond_async(self, user_input: str, session_id: str | None = None) -> str:
        """Async response with function calling."""
        
        ACTIVE_SESSIONS.inc()
        try:
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
                with tracer.start_as_current_span("llm.generation") as span:
                    span.set_attribute("llm.model", self.llm.model_name)
                    start_time = time.time()
                    llm_response = await self.llm.generate_with_tools_async(messages, OPENAI_TOOLS)
                    
                    duration = time.time() - start_time
                    LLM_LATENCY.labels(model_name=self.llm.model_name).observe(duration)
                    span.set_attribute("llm.latency_seconds", duration)
                    
                    if "usage" in llm_response:
                        TOKEN_USAGE.labels(type="prompt").inc(llm_response["usage"].get("prompt_tokens", 0))
                        TOKEN_USAGE.labels(type="completion").inc(llm_response["usage"].get("completion_tokens", 0))

                log_llm_call(self.llm.model_name, success=True, context=ctx)

                # Check if LLM wants to call tools
                if llm_response.get("tool_calls"):
                    response = await self._handle_tool_calls_async(llm_response, messages, ctx)
                else:
                    response = llm_response.get("reply") or "I'm not sure how to respond to that."

            except Exception as error:
                log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
                response = get_fallback_response("llm_unavailable", details=str(error))
                log_fallback("llm_unavailable", details=str(error), context=ctx)

            self.add_message("assistant", response)
            log_response(response, ctx)

            return response
        finally:
            ACTIVE_SESSIONS.dec()

    async def _handle_tool_calls_async(self, llm_response: dict, messages: list, ctx) -> str:
        """Async tool call handling."""

        iterations = 0
        MAX_ITERATIONS = 5

        while llm_response.get("tool_calls") and iterations < MAX_ITERATIONS:
            iterations += 1
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
                with tracer.start_as_current_span("llm.generation") as span:
                    span.set_attribute("llm.model", self.llm.model_name)
                    start_time = time.time()
                    
                    # Use generate_with_context_async if available (for Ollama), otherwise generate_with_tools_async
                    if hasattr(self.llm, 'generate_with_context_async'):
                        llm_response = await self.llm.generate_with_context_async(messages)
                    else:
                        llm_response = await self.llm.generate_with_tools_async(messages, OPENAI_TOOLS)
                        
                    duration = time.time() - start_time
                    LLM_LATENCY.labels(model_name=self.llm.model_name).observe(duration)
                    span.set_attribute("llm.latency_seconds", duration)
                    
                    if "usage" in llm_response:
                        TOKEN_USAGE.labels(type="prompt").inc(llm_response["usage"].get("prompt_tokens", 0))
                        TOKEN_USAGE.labels(type="completion").inc(llm_response["usage"].get("completion_tokens", 0))

                log_llm_call(self.llm.model_name, success=True, context=ctx)

            except Exception as error:
                log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)
                return get_fallback_response("tool_failed", details=str(error))

        if iterations >= MAX_ITERATIONS:
            log_event("max_iterations_reached", {"context": ctx})
            return "I needed too many steps to solve this problem. Please try rephrasing your request."

        return llm_response.get("reply") or "I'm not sure how to respond to that."

    async def _execute_single_tool_async(self, tool_call: dict, ctx) -> dict:
        """Execute a single tool asynchronously."""

        tool_name = tool_call["tool"] 

        try:
            arguments = json.loads(tool_call["arguments"])
        except json.JSONDecodeError:
            arguments = {}

        log_tool_planned({"tool": tool_name, "arguments": arguments}, ctx)

        # Execute tool (tools are sync, but we run them in thread pool)
        try:
            tool_result = await asyncio.wait_for(
                asyncio.to_thread(
                    execute_tool,
                    {"tool": tool_name, "arguments": arguments},
                    user_plan=self.metadata.get("user_plan", get_config().USER_PLAN)
                ),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            tool_result = {
                "status": "error",
                "message": f"Tool '{tool_name}' timed out after 15.0 seconds."
            }

        
        success = tool_result.get("status") == "success"

        log_tool_result(
            tool_name=tool_name,
            success=success,
            result=tool_result,
            context=ctx
        )

        return tool_result

    async def respond_stream_async(self, user_input: str, session_id: str | None = None):
        """Async streaming response."""
        ACTIVE_SESSIONS.inc()
        try:
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
                collected_content: list[str] = []
                iterations = 0
                MAX_ITERATIONS = 5

                while iterations < MAX_ITERATIONS:
                    iterations += 1
                    tool_calls_data: list[dict] | None = None
                    iteration_content: list[str] = []

                    with tracer.start_as_current_span("llm.generation") as span:
                        span.set_attribute("llm.model", self.llm.model_name)
                        start_time = time.time()
                        
                        async for chunk in self.llm.generate_with_tools_stream_async(messages, OPENAI_TOOLS):  # type: ignore[union-attr]
                            if chunk["type"] == "content":
                                iteration_content.append(chunk["content"])
                                collected_content.append(chunk["content"])
                                yield chunk["content"]
                            elif chunk["type"] == "tool_calls":
                                tool_calls_data = chunk["tool_calls"]
                                
                        duration = time.time() - start_time
                        LLM_LATENCY.labels(model_name=self.llm.model_name).observe(duration)
                        span.set_attribute("llm.latency_seconds", duration)

                    log_llm_call(self.llm.model_name, success=True, context=ctx)

                    if not tool_calls_data:
                        break

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
                        "content": "".join(iteration_content),
                        "tool_calls": formatted_tool_calls  # type: ignore[dict-item]
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

                if iterations >= MAX_ITERATIONS:
                    log_event("max_iterations_reached", {"context": ctx})
                    fallback = "I needed too many steps to solve this problem. Please try rephrasing your request."
                    collected_content.append(fallback)
                    yield fallback

                full_response = "".join(collected_content)

                self.add_message("assistant", full_response)
                log_response(full_response, ctx)

            except Exception as error:
                log_llm_call(self.llm.model_name, success=False, error=str(error), context=ctx)

                fallback = get_fallback_response("llm_unavailable", details=str(error))
                log_fallback("llm_unavailable", details=str(error), context=ctx)

                self.add_message("assistant", fallback)

                yield fallback
        finally:
            ACTIVE_SESSIONS.dec()