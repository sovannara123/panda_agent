from abc import ABC, abstractmethod
from openai import OpenAI, AsyncOpenAI
import requests
import json
import asyncio
import httpx
from typing import AsyncGenerator, Generator
from config import get_config
from logger import get_logger, log_event


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class RetryError(LLMError):
    """Raised when max retry attempts exceeded."""
    pass


class LLMClient(ABC):
    """Base interface for all LLM providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def generate(self, prompt: str) -> dict:
        """Generate response from prompt."""
        pass

    @abstractmethod
    def generate_with_tools(self, messages: list, tools: list) -> dict:
        """Generate response with tool calling support."""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str) -> Generator[dict, None, None]:
        """Stream response token by token. Yields content chunks."""
        pass

    @abstractmethod
    def generate_with_tools_stream(self, messages: list, tools: list) -> Generator[dict, None, None]:
        """Stream response with function calling. Yields content chunks and tool calls."""
        pass

    @abstractmethod
    async def generate_async(self, prompt: str) -> dict:
        """Async generate response from prompt."""
        pass

    @abstractmethod
    async def generate_with_tools_async(self, messages: list, tools: list) -> dict:
        """Async generate response with tool calling support."""
        pass

    @abstractmethod
    async def generate_stream_async(self, prompt: str) -> AsyncGenerator[dict, None]:
        """Async stream response token by token. Yields content chunks."""
        pass

    @abstractmethod
    async def generate_with_tools_stream_async(self, messages: list, tools: list) -> AsyncGenerator[dict, None]:
        """Async stream response with function calling support."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI GPT adapter."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None, base_url: str | None = None):
        self._model_name = model_name or get_config().MODEL_NAME
        self.api_key = api_key or get_config().API_KEY

        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set API_KEY in .env")

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> dict:
        """Simple text generation."""
        try:
            response = self.client.chat.completions.create(
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
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

        except Exception as error:
            log_event("openai_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    def generate_with_tools(self, messages: list, tools: list) -> dict:
        """Generate with function calling."""
        try:
            response = self.client.chat.completions.create(
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
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "tool": tool_call.function.name,  # type: ignore[attr-defined]
                        "arguments": tool_call.function.arguments  # type: ignore[attr-defined]
                    })

            return result

        except Exception as error:
            log_event("openai_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    def generate_stream(self, prompt: str):
        """Stream response token by token."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                stream=True
            )

            collected_content = []
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_content.append(content)
                    yield {
                        "type": "content",
                        "content": content
                    }

            yield {
                "type": "done",
                "full_content": "".join(collected_content),
                "usage": None  # Usage not available in streaming mode
            }

        except Exception as error:
            log_event("openai_stream_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    def generate_with_tools_stream(self, messages: list, tools: list):
        """Stream response with function calling support."""
        try:
            stream = self.client.chat.completions.create(
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

            for chunk in stream:
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
                "full_content": "".join(collected_content),
                "usage": None
            }

        except Exception as error:
            log_event("openai_stream_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    async def generate_async(self, prompt: str) -> dict:
        """Async simple text generation."""
        try:
            response = await self._async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )

            return {
                "model": self.model_name,
                "reply": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

        except Exception as error:
            log_event("openai_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    async def generate_with_tools_async(self, messages: list, tools: list) -> dict:
        """Async generate with function calling."""
        try:
            response = await self._async_client.chat.completions.create(
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
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "tool": tool_call.function.name,  # type: ignore[attr-defined]
                        "arguments": tool_call.function.arguments  # type: ignore[attr-defined]
                    })

            return result

        except Exception as error:
            log_event("openai_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    async def generate_stream_async(self, prompt: str) -> AsyncGenerator[dict, None]:
        """Async stream response token by token."""
        try:
            stream = await self._async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
                stream=True
            )

            collected_content = []
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_content.append(content)
                    yield {
                        "type": "content",
                        "content": content
                    }

            yield {
                "type": "done",
                "full_content": "".join(collected_content),
                "usage": None
            }

        except Exception as error:
            log_event("openai_stream_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    async def generate_with_tools_stream_async(self, messages: list, tools: list) -> AsyncGenerator[dict, None]:
        """Async stream response with function calling support."""
        try:
            stream = await self._async_client.chat.completions.create(
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
                "full_content": "".join(collected_content),
                "usage": None
            }

        except Exception as error:
            log_event("openai_stream_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise


class GroqClient(LLMClient):
    """Groq adapter (OpenAI-compatible)."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        self._model_name = model_name or get_config().MODEL_NAME
        self.api_key = api_key or get_config().GROQ_API_KEY

        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )

            return {
                "model": self.model_name,
                "reply": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

        except Exception as error:
            log_event("groq_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    def generate_with_tools(self, messages: list, tools: list) -> dict:
        try:
            response = self.client.chat.completions.create(
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
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "tool": tool_call.function.name,  # type: ignore[attr-defined]
                        "arguments": tool_call.function.arguments  # type: ignore[attr-defined]
                    })

            return result

        except Exception as error:
            log_event("groq_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    def generate_stream(self, prompt: str):
        """Stream response token by token."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
                stream=True
            )

            collected_content = []
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_content.append(content)
                    yield {
                        "type": "content",
                        "content": content
                    }

            yield {
                "type": "done",
                "full_content": "".join(collected_content),
                "usage": None
            }

        except Exception as error:
            log_event("groq_stream_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    def generate_with_tools_stream(self, messages: list, tools: list):
        """Stream response with function calling support."""
        try:
            stream = self.client.chat.completions.create(
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

            for chunk in stream:
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
                "full_content": "".join(collected_content),
                "usage": None
            }

        except Exception as error:
            log_event("groq_stream_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    async def generate_async(self, prompt: str) -> dict:
        """Async simple text generation."""
        try:
            response = await self._async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )

            return {
                "model": self.model_name,
                "reply": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

        except Exception as error:
            log_event("groq_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    async def generate_with_tools_async(self, messages: list, tools: list) -> dict:
        """Async generate with function calling."""
        try:
            response = await self._async_client.chat.completions.create(
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
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "tool": tool_call.function.name,  # type: ignore[attr-defined]
                        "arguments": tool_call.function.arguments  # type: ignore[attr-defined]
                    })

            return result

        except Exception as error:
            log_event("groq_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    async def generate_stream_async(self, prompt: str) -> AsyncGenerator[dict, None]:
        """Async stream response token by token."""
        try:
            stream = await self._async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
                stream=True
            )

            collected_content = []
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_content.append(content)
                    yield {
                        "type": "content",
                        "content": content
                    }

            yield {
                "type": "done",
                "full_content": "".join(collected_content),
                "usage": None
            }

        except Exception as error:
            log_event("groq_stream_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    async def generate_with_tools_stream_async(self, messages: list, tools: list) -> AsyncGenerator[dict, None]:
        """Async stream response with function calling support."""
        try:
            stream = await self._async_client.chat.completions.create(
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
                "full_content": "".join(collected_content),
                "usage": None
            }

        except Exception as error:
            log_event("groq_stream_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise


class MockLLMClient(LLMClient):
    """Mock client for testing."""

    def __init__(self, model_name: str = "mock-model"):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> dict:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if "crash" in prompt.lower():
            raise Exception("Simulated LLM API crash.")

        return {
            "model": self.model_name,
            "reply": f"[Mock LLM] I received your request.",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }

    def generate_with_tools(self, messages: list, tools: list) -> dict:
        return {
            "model": self.model_name,
            "reply": None,
            "tool_calls": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }

    def generate_stream(self, prompt: str):
        """Mock streaming - yields full response in chunks."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if "crash" in prompt.lower():
            raise Exception("Simulated LLM API crash.")

        reply = f"[Mock LLM] I received your request."
        words = reply.split()

        for i, word in enumerate(words):
            if i > 0:
                yield {"type": "content", "content": " " + word}
            else:
                yield {"type": "content", "content": word}

        yield {
            "type": "done",
            "full_content": reply,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }

    def generate_with_tools_stream(self, messages: list, tools: list):
        """Mock streaming with tools - no tool calls, just content."""
        reply = None
        words = "Mock response with tools.".split()

        for i, word in enumerate(words):
            if i > 0:
                yield {"type": "content", "content": " " + word}
            else:
                yield {"type": "content", "content": word}

        yield {
            "type": "done",
            "full_content": "Mock response with tools.",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }

    async def generate_async(self, prompt: str) -> dict:
        """Async mock generation."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if "crash" in prompt.lower():
            raise Exception("Simulated LLM API crash.")

        reply = f"[Mock LLM] I received your request."
        return {
            "model": self.model_name,
            "reply": reply,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }

    async def generate_with_tools_async(self, messages: list, tools: list) -> dict:
        """Async mock generate with tools."""
        return {
            "model": self.model_name,
            "reply": None,
            "tool_calls": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }

    async def generate_stream_async(self, prompt: str) -> AsyncGenerator[dict, None]:
        """Async mock streaming - yields full response in chunks."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if "crash" in prompt.lower():
            raise Exception("Simulated LLM API crash.")

        reply = f"[Mock LLM] I received your request."
        words = reply.split()

        for i, word in enumerate(words):
            if i > 0:
                yield {"type": "content", "content": " " + word}
            else:
                yield {"type": "content", "content": word}
            await asyncio.sleep(0.01)  # Simulate streaming delay

        yield {
            "type": "done",
            "full_content": reply,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }

    async def generate_with_tools_stream_async(self, messages: list, tools: list) -> AsyncGenerator[dict, None]:
        """Async mock streaming with tools - no tool calls, just content."""
        reply = "Mock response with tools."
        words = reply.split()

        for i, word in enumerate(words):
            if i > 0:
                yield {"type": "content", "content": " " + word}
            else:
                yield {"type": "content", "content": word}
            await asyncio.sleep(0.01)

        yield {
            "type": "done",
            "full_content": reply,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }


class FlakyMockLLMClient(MockLLMClient):
    """Mock client that fails N times then succeeds."""

    def __init__(self, model_name: str = "mock-model", fail_times: int = 2):
        super().__init__(model_name)
        self.fail_times = fail_times
        self.attempts = 0

    def generate(self, prompt: str) -> dict:
        self.attempts += 1

        if self.attempts <= self.fail_times:
            raise LLMError(f"Simulated LLM failure on attempt {self.attempts}.")

        return super().generate(prompt)

    def generate_with_tools(self, messages: list, tools: list) -> dict:
        self.attempts += 1

        if self.attempts <= self.fail_times:
            raise LLMError(f"Simulated LLM failure on attempt {self.attempts}.")

        return super().generate_with_tools(messages, tools)

    async def generate_async(self, prompt: str) -> dict:
        """Async generate with failure simulation."""
        self.attempts += 1

        if self.attempts <= self.fail_times:
            raise LLMError(f"Simulated LLM failure on attempt {self.attempts}.")

        return await super().generate_async(prompt)

    async def generate_with_tools_async(self, messages: list, tools: list) -> dict:
        """Async generate with tools and failure simulation."""
        self.attempts += 1

        if self.attempts <= self.fail_times:
            raise LLMError(f"Simulated LLM failure on attempt {self.attempts}.")

        return await super().generate_with_tools_async(messages, tools)

    async def generate_stream_async(self, prompt: str) -> AsyncGenerator[dict, None]:
        """Async stream with failure simulation."""
        self.attempts += 1

        if self.attempts <= self.fail_times:
            raise LLMError(f"Simulated LLM failure on attempt {self.attempts}.")

        async for chunk in super().generate_stream_async(prompt):
            yield chunk

    async def generate_with_tools_stream_async(self, messages: list, tools: list) -> AsyncGenerator[dict, None]:
        """Async stream with tools and failure simulation."""
        self.attempts += 1

        if self.attempts <= self.fail_times:
            raise LLMError(f"Simulated LLM failure on attempt {self.attempts}.")

        async for chunk in super().generate_with_tools_stream_async(messages, tools):
            yield chunk



class OllamaClient(LLMClient):
    """Ollama local LLM adapter."""

    def __init__(self, model_name: str | None = None, host: str | None = None):
        self._model_name = model_name or get_config().MODEL_NAME
        self.host = host or get_config().OLLAMA_HOST

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> dict:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        url = f"{self.host.rstrip('/')}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
        except requests.RequestException as error:
            log_event("ollama_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

        data = response.json()
        reply = data.get("response", "").strip()

        if not reply:
            raise Exception("Ollama returned an empty response.")

        return {
            "model": self.model_name,
            "reply": reply,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    def generate_with_tools(self, messages: list, tools: list) -> dict:
        # Ollama doesn't support native tool calling
        # Fall back to simple generation with last user message
        last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        result = self.generate(last_user_msg)
        result["tool_calls"] = []
        return result

    def generate_stream(self, prompt: str):
        """Stream response from Ollama."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        url = f"{self.host.rstrip('/')}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True
        }

        try:
            response = requests.post(url, json=payload, timeout=120, stream=True)
            response.raise_for_status()
        except requests.RequestException as error:
            log_event("ollama_stream_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

        collected_content = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            if "response" in data:
                chunk = data["response"]
                if chunk:
                    collected_content.append(chunk)
                    yield {
                        "type": "content",
                        "content": chunk
                    }

            if data.get("done", False):
                break

        yield {
            "type": "done",
            "full_content": "".join(collected_content),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    def generate_with_tools_stream(self, messages: list, tools: list):
        """Ollama doesn't support native tool calling - fallback to simple stream."""
        last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        for chunk in self.generate_stream(last_user_msg):
            yield chunk
        yield {
            "type": "tool_calls",
            "tool_calls": []
        }

    async def generate_async(self, prompt: str) -> dict:
        """Async generate using Ollama API."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        url = f"{self.host.rstrip('/')}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except httpx.RequestError as error:
            log_event("ollama_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

        data = response.json()
        reply = data.get("response", "").strip()

        if not reply:
            raise Exception("Ollama returned an empty response.")

        return {
            "model": self.model_name,
            "reply": reply,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    async def generate_with_tools_async(self, messages: list, tools: list) -> dict:
        """Async generate with tools - Ollama doesn't support native tool calling."""
        last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        result = await self.generate_async(last_user_msg)
        result["tool_calls"] = []
        return result

    async def generate_stream_async(self, prompt: str) -> AsyncGenerator[dict, None]:
        """Async stream response from Ollama."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        url = f"{self.host.rstrip('/')}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    collected_content = []
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if "response" in data:
                            chunk = data["response"]
                            if chunk:
                                collected_content.append(chunk)
                                yield {
                                    "type": "content",
                                    "content": chunk
                                }

                        if data.get("done", False):
                            break

                    yield {
                        "type": "done",
                        "full_content": "".join(collected_content),
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }

        except httpx.RequestError as error:
            log_event("ollama_stream_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise

    async def generate_with_tools_stream_async(self, messages: list, tools: list) -> AsyncGenerator[dict, None]:
        """Ollama doesn't support native tool calling - fallback to simple stream."""
        last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        async for chunk in self.generate_stream_async(last_user_msg):
            yield chunk
        yield {
            "type": "tool_calls",
            "tool_calls": []
        }


def create_llm_client(provider: str | None = None, model_name: str | None = None, api_key: str | None = None) -> LLMClient:
    """Factory function to create LLM client based on provider."""
    provider = (provider or get_config().LLM_PROVIDER).lower()

    if provider == "openai":
        return OpenAIClient(
            model_name=model_name or get_config().MODEL_NAME,
            api_key=api_key or get_config().API_KEY
        )

    if provider == "groq":
        return GroqClient(
            model_name=model_name or get_config().MODEL_NAME,
            api_key=api_key or get_config().GROQ_API_KEY
        )

    if provider == "mock":
        return MockLLMClient(model_name or get_config().MODEL_NAME)

    if provider == "ollama":
        return OllamaClient(model_name=model_name or get_config().MODEL_NAME)

    raise ValueError(f"Unknown LLM provider: {provider}")