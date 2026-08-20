import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    AGENT_NAME = os.getenv("AGENT_NAME", "Panda")
    AGENT_ROLE = os.getenv("AGENT_ROLE", "AI learning assistant")
    AGENT_VERSION = os.getenv("AGENT_VERSION", "1.0")

    API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    MESSAGE_LIMIT_FREE = int(os.getenv("MESSAGE_LIMIT_FREE", "10"))
    MESSAGE_LIMIT_PREMIUM = None  # None = unlimited

    MAX_HISTORY = 5 