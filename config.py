import os

from dotenv import load_dotenv

load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "Nova")
AGENT_ROLE = os.getenv("AGENT_ROLE", "AI learning assistant")
AGENT_VERSION = os.getenv("AGENT_VERSION", "1.0")

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

MESSAGE_LIMIT_FREE = int(os.getenv("MESSAGE_LIMIT_FREE", "10"))
MESSAGE_LIMIT_PREMIUM = None  # None = unlimited