import logging

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class MockLLMClient:
    def __init__(self, model_name="mock-model"):
        self.model_name = model_name

    def generate(self, prompt):
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt rejected")
            raise LLMError("Prompt cannot be empty.")

        if "crash" in prompt.lower():
            logger.warning("Simulated LLM crash triggered")
            raise LLMError("Simulated LLM API crash.")

        logger.info("LLM response generated with model %s", self.model_name)
        return {
            "model": self.model_name,
            "reply": f"[Mock LLM] I received your request."
        }