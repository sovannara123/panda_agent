class LLMError(Exception):
    pass


class MockLLMClient:
    def __init__(self, model_name="mock-model"):
        self.model_name = model_name

    def generate(self, prompt):
        if not prompt or not prompt.strip():
            raise LLMError("Prompt cannot be empty.")

        if "crash" in prompt.lower():
            raise LLMError("Simulated LLM API crash.")

        return {
            "model": self.model_name,
            "reply": f"[Mock LLM] I received your request."
        }