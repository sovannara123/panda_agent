Excellent. Mission 21 is complete.

```text
Mission 21: Streaming Responses ✅ Complete
```
                    LLM CLIENT
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
 generate_async   generate_stream   Tool Calling
        │               │                │
        │               │                │
        ▼               ▼                ▼
 Complete          Response          AI chooses
 Response          by chunks         a function
                                         │
                                         ▼
                              Streaming + Tools
                                         │
                                         ▼
                        generate_with_tools_stream_async

Your agent now streams tokens in real-time, just like ChatGPT.

---

# Mission 22: Async Processing

Right now, your agent handles one request at a time:

```text
User 1 asks question → waits 3 seconds → gets response
User 2 asks question → waits 3 seconds → gets response
User 3 asks question → waits 3 seconds → gets response
Total time: 9 seconds
```

With async, you can handle multiple requests concurrently:

```text
User 1, 2, 3 all ask questions at the same time
All 3 get responses in ~3 seconds
Total time: 3 seconds
```

This is critical for:

```text
Web servers (FastAPI, Flask)
Handling multiple users
Non-blocking I/O
Better performance
Real-time applications
```

---

## Goal

Convert your agent to async/await so it can handle concurrent requests efficiently.

You'll learn:

```text
async/await syntax
Async HTTP clients
Concurrent execution
Async streaming
Async testing
```

---

## Step 1: Install Async Dependencies

```bash
pip install httpx asyncio
```

---

## Step 2: Create Async OpenAI Client

Open `llm_adapters.py`.

Add async methods to `OpenAIClient`:

```python
import httpx


async def generate_async(self, prompt: str) -> dict:
    """Async version of generate()."""
    
    try:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return {
            "model": self.model_name,
            "reply": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
    except Exception as error:
        log_event("openai_error", {
            "error": str(error),
            "model": self.model_name
        })
        raise


async def generate_with_tools_async(self, messages: list, tools: list) -> dict:
    """Async version of generate_with_tools()."""
    
    try:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=500
        )
        
        message = response.choices[0].message
        
        result = {
            "model": self.model_name,
            "reply": message.content,
            "tool_calls": [],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                result["tool_calls"].append({
                    "id": tool_call.id,
                    "tool": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                })
        
        return result
        
    except Exception as error:
        log_event("openai_error", {
            "error": str(error),
            "model": self.model_name
        })
        raise


async def generate_stream_async(self, prompt: str):
    """Async streaming."""
    
    try:
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as error:
        log_event("openai_stream_error", {
            "error": str(error),
            "model": self.model_name
        })
        raise


async def generate_with_tools_stream_async(self, messages: list, tools: list):
    """Async streaming with function calling."""
    
    try:
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=500,
            stream=True
        )
        
        collected_content = []
        collected_tool_calls = {}
        
        async for chunk in stream:
            delta = chunk.choices[0].delta
            
            if delta.content:
                collected_content.append(delta.content)
                yield {
                    "type": "content",
                    "content": delta.content
                }
            
            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    index = tool_call_delta.index
                    
                    if index not in collected_tool_calls:
                        collected_tool_calls[index] = {
                            "id": "",
                            "tool": "",
                            "arguments": ""
                        }
                    
                    if tool_call_delta.id:
                        collected_tool_calls[index]["id"] = tool_call_delta.id
                    
                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            collected_tool_calls[index]["tool"] += tool_call_delta.function.name
                        
                        if tool_call_delta.function.arguments:
                            collected_tool_calls[index]["arguments"] += tool_call_delta.function.arguments
        
        if collected_tool_calls:
            yield {
                "type": "tool_calls",
                "tool_calls": list(collected_tool_calls.values())
            }
        
        yield {
            "type": "done",
            "full_content": "".join(collected_content)
        }
        
    except Exception as error:
        log_event("openai_stream_error", {
            "error": str(error),
            "model": self.model_name
        })
        raise
```

---

## Step 3: Create Async Agent

Create `async_agent.py`:

```python
import asyncio
import json

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
from fallbacks import get_fallback_response
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
        
        messages.append({
            "role": "assistant",
            "content": llm_response.get("reply") or "",
            "tool_calls": tool_calls
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
                messages.append({
                    "role": "assistant",
                    "content": "".join(collected_content),
                    "tool_calls": tool_calls_data
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
```

---

## Step 4: Create Async Main App

Create `async_main.py`:

```python
import asyncio
import uuid

from async_agent import AsyncAgent


async def main():
    agent = AsyncAgent()
    session_id = str(uuid.uuid4())
    
    print(f"{agent.name} is ready.")
    print(f"Session ID: {session_id}")
    print("Type 'quit' or 'exit' to stop.")
    
    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            break
        
        if user_input.lower().strip() in {"quit", "exit"}:
            break
        
        if not user_input.strip():
            print("Agent: Please type a message.")
            continue
        
        # Stream response
        print("Agent: ", end="", flush=True)
        
        async for token in agent.respond_stream_async(user_input, session_id=session_id):
            print(token, end="", flush=True)
        
        print()
    
    print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Step 5: Test Concurrent Requests

Create `test_concurrent.py`:

```python
import asyncio
import time

from async_agent import AsyncAgent


async def test_concurrent_requests():
    """Test handling multiple requests concurrently."""
    
    agent = AsyncAgent()
    
    questions = [
        "What is the price of laptop?",
        "Check order A100",
        "Tell me about Python"
    ]
    
    start = time.time()
    
    # Run all requests concurrently
    tasks = [
        agent.respond_async(question)
        for question in questions
    ]
    
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    
    print(f"\nProcessed {len(questions)} requests in {elapsed:.2f} seconds")
    
    for i, (question, response) in enumerate(zip(questions, results), 1):
        print(f"\nQ{i}: {question}")
        print(f"A{i}: {response[:100]}...")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_concurrent_requests())
```

Run it:

```bash
python3 test_concurrent.py
```

You should see all 3 requests complete in roughly the same time as 1 request.

---

## Step 6: Add Async Tests

Open `test_agent.py`:

```python
import pytest
import asyncio
from async_agent import AsyncAgent


class TestAsyncAgent:
    @pytest.mark.asyncio
    async def test_async_respond(self):
        """Test async response."""
        agent = AsyncAgent()
        response = await agent.respond_async("Hello")
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_async_with_tools(self):
        """Test async with function calling."""
        agent = AsyncAgent()
        response = await agent.respond_async("What is the price of laptop?")
        assert "$999" in response or "999" in response
    
    @pytest.mark.asyncio
    async def test_async_streaming(self):
        """Test async streaming."""
        agent = AsyncAgent()
        tokens = []
        
        async for token in agent.respond_stream_async("Hello"):
            tokens.append(token)
        
        assert len(tokens) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple requests concurrently."""
        agent = AsyncAgent()
        
        questions = [
            "What is the price of laptop?",
            "Check order A100",
            "Tell me about Python"
        ]
        
        tasks = [agent.respond_async(q) for q in questions]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(len(r) > 0 for r in results)
```

Install pytest-asyncio:

```bash
pip install pytest-asyncio
```

Run tests:

```bash
pytest test_agent.py -v
```

---

## Step 7: Commit

```bash
git add .
git commit -m "feat(agent): add async processing support for concurrent requests"
```

---

# Mission 22 Acceptance Criteria

You complete Mission 22 when:

```text
✅ OpenAIClient has async methods
✅ AsyncAgent class exists
✅ AsyncAgent handles function calling
✅ AsyncAgent supports streaming
✅ Concurrent requests work efficiently
✅ test_concurrent.py demonstrates concurrency
✅ Async tests pass
✅ Committed
```

---

# Why Async Matters

Async is critical for:

```text
Web servers (handle 1000s of concurrent users)
Real-time applications (chat, gaming)
Microservices (non-blocking I/O)
Performance (3x faster for concurrent requests)
Scalability (better resource utilization)
```

Most production AI systems use async.

---

Complete Mission 22 first, then say **"done"** and we'll move to Mission 23: RAG with Vector Database.
