import logging

import requests

from config import Config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class RetryError(LLMError):
    pass


class LLMClient:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, prompt):
        raise NotImplementedError

    def _validate_prompt(self, prompt):
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt rejected")
            raise LLMError("Prompt cannot be empty.")


class MockLLMClient(LLMClient):
    def __init__(self, model_name="mock-model"):
        super().__init__(model_name)

    def generate(self, prompt):
        self._validate_prompt(prompt)

        if "crash" in prompt.lower():
            logger.warning("Simulated LLM crash triggered")
            raise LLMError("Simulated LLM API crash.")

        logger.info("LLM response generated with model %s", self.model_name)
        return {
            "model": self.model_name,
            "reply": f"[Mock LLM] I received your request."
        }


class OpenAILLMClient(LLMClient):
    def __init__(self, api_key, model_name):
        super().__init__(model_name)
        self.api_key = api_key

    def generate(self, prompt):
        self._validate_prompt(prompt)

        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("LLM API request failed")
            raise LLMError("LLM API request failed.")

        data = response.json()

        logger.info("LLM response generated with model %s", self.model_name)
        return {
            "model": self.model_name,
            "reply": data["choices"][0]["message"]["content"]
        }


class GeminiLLMClient(LLMClient):
    def __init__(self, api_key, model_name):
        super().__init__(model_name)
        self.api_key = api_key

    def generate(self, prompt):
        self._validate_prompt(prompt)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("LLM API request failed")
            raise LLMError("LLM API request failed.")

        data = response.json()

        logger.info("LLM response generated with model %s", self.model_name)
        reply = data["candidates"][0]["content"]["parts"][0]["text"]
        return {
            "model": self.model_name,
            "reply": reply
        }


def create_llm_client():
    provider = Config.LLM_PROVIDER.lower()

    if provider == "openai":
        if not Config.API_KEY:
            logger.error("LLM_PROVIDER is openai but OPENAI_API_KEY is not set")
            raise LLMError("OPENAI_API_KEY is required for the OpenAI provider.")
        return OpenAILLMClient(Config.API_KEY, Config.MODEL_NAME)

    if provider == "gemini":
        if not Config.API_KEY:
            logger.error("LLM_PROVIDER is gemini but GEMINI_API_KEY is not set")
            raise LLMError("GEMINI_API_KEY is required for the Gemini provider.")
        return GeminiLLMClient(Config.API_KEY, Config.MODEL_NAME)

    if provider == "mock":
        return MockLLMClient(Config.MODEL_NAME)

    logger.error("Unknown LLM provider: %s", provider)
    raise LLMError(f"Unknown LLM provider: {provider}")
