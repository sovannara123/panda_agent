from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from openai import OpenAI
from config import Config
from logger import log_event


class LLMClient(ABC):
    """Base interface for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> dict:
        """Generate response from prompt."""
        pass

    @abstractmethod
    def generate_with_tools(self, messages: list, tools: list) -> dict:
        """Generate response with tool calling support."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI GPT adapter."""

    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        self.model_name = model_name or Config.MODEL_NAME
        self.api_key = api_key or Config.API_KEY

        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set API_KEY in .env")

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

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


class GroqClient(LLMClient):
    """Groq adapter (OpenAI-compatible)."""

    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name = model_name or Config.MODEL_NAME
        self.api_key = api_key or Config.GROQ_API_KEY

        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )

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
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
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
            log_event("groq_error", {
                "error": str(error),
                "model": self.model_name
            })
            raise


class MockLLMClient(LLMClient):
    """Mock client for testing."""

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name

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


class FlakyMockLLMClient(MockLLMClient):
    """Mock client that fails N times then succeeds."""

    def __init__(self, model_name: str = "mock-model", fail_times: int = 2):
        super().__init__(model_name)
        self.fail_times = fail_times
        self.attempts = 0

    def generate(self, prompt: str) -> dict:
        self.attempts += 1

        if self.attempts <= self.fail_times:
            raise Exception(f"Simulated LLM failure on attempt {self.attempts}.")

        return super().generate(prompt)

    def generate_with_tools(self, messages: list, tools: list) -> dict:
        self.attempts += 1

        if self.attempts <= self.fail_times:
            raise Exception(f"Simulated LLM failure on attempt {self.attempts}.")

        return super().generate_with_tools(messages, tools)


def create_llm_client(provider: str = None, model_name: str = None, api_key: str = None) -> LLMClient:
    """Factory function to create LLM client based on provider."""
    provider = (provider or Config.LLM_PROVIDER).lower()

    if provider == "openai":
        return OpenAIClient(
            model_name=model_name or Config.MODEL_NAME,
            api_key=api_key or Config.API_KEY
        )

    if provider == "groq":
        return GroqClient(
            model_name=model_name or Config.MODEL_NAME,
            api_key=api_key or Config.GROQ_API_KEY
        )

    if provider == "mock":
        return MockLLMClient(model_name or Config.MODEL_NAME)

    raise ValueError(f"Unknown LLM provider: {provider}")