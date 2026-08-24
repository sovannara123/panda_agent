import logging
import requests


from config import Config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class RetryError(LLMError):
    pass

#ASTRACT BASE 
class LLMClient:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, prompt):
        raise NotImplementedError

    def _validate_prompt(self, prompt):
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt rejected")
            raise LLMError("Prompt cannot be empty.")

#TESTING FALLBACK 
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
#SIMULATE THE FAILURE FOR RETRY TESTING 
class FlakyMockLLMClient(MockLLMClient):
    def __init__(self, model_name="mock-model", fail_times: int = 2):
        super().__init__(model_name)
        self.fail_times = fail_times
        self.attempts = 0

    def generate(self, prompt: str):
        self.attempts += 1

        if self.attempts <= self.fail_times:
            raise LLMError(
                f"Simulated LLM failure on attempt {self.attempts}."
            )

        return super().generate(prompt)

#OPENAI API 
class OpenAILLMClient(LLMClient):
    def __init__(self, api_key, model_name, base_url="https://api.openai.com/v1"):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, prompt):
        self._validate_prompt(prompt)

        url = f"{self.base_url}/chat/completions"

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

#LOCAL OLLAMA SERVER 
class OllamaLLMClient(LLMClient):
    def __init__(self, model_name, host=None):
        super().__init__(model_name)
        self.host = host or Config.OLLAMA_HOST

    def generate(self, prompt):
        self._validate_prompt(prompt)

        url = f"{self.host.rstrip('/')}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("Ollama API request failed")
            raise LLMError("Ollama API request failed.")

        data = response.json()

        reply = data.get("response", "").strip()
        if not reply:
            logger.warning("Ollama returned an empty response")
            raise LLMError("Ollama returned an empty response.")

        logger.info("LLM response generated with model %s", self.model_name)
        return {
            "model": self.model_name,
            "reply": reply
        }

#GEMINI API 

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

    if provider == "groq":
        if not Config.GROQ_API_KEY:
            logger.error("LLM_PROVIDER is groq but GROQ_API_KEY is not set")
            raise LLMError("GROQ_API_KEY is required for the Groq provider.")
        return OpenAILLMClient(
            Config.GROQ_API_KEY,
            Config.MODEL_NAME,
            base_url="https://api.groq.com/openai/v1"
        )

    if provider == "gemini":
        if not Config.API_KEY:
            logger.error("LLM_PROVIDER is gemini but GEMINI_API_KEY is not set")
            raise LLMError("GEMINI_API_KEY is required for the Gemini provider.")
        return GeminiLLMClient(Config.API_KEY, Config.MODEL_NAME)

    if provider == "mock":
        return MockLLMClient(Config.MODEL_NAME)

    if provider == "ollama":
        return OllamaLLMClient(Config.MODEL_NAME)

    logger.error("Unknown LLM provider: %s", provider)
    raise LLMError(f"Unknown LLM provider: {provider}")
