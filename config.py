import os
from pydantic import Field, model_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Agent Configuration
    AGENT_NAME: str = "NARA"
    AGENT_ROLE: str = "AI learning assistant"
    AGENT_VERSION: str = "1.0"

    # LLM Configuration
    LLM_PROVIDER: str = Field(default="mock", pattern="^(mock|openai|groq|ollama|gemini)$")
    MODEL_NAME: str = "gpt-4o-mini"
    OLLAMA_HOST: str = "http://localhost:11434"

    # Embedding Configuration
    EMBEDDING_PROVIDER: str = Field(default="local", pattern="^(local|openai)$")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # API Keys (optional, validated at runtime based on provider)
    API_KEY: str = ""
    GROQ_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None

    # Rate Limiting
    MESSAGE_LIMIT_FREE: int = Field(default=10, ge=1, le=10000)
    USER_PLAN: str = Field(default="free", pattern="^(free|premium)$") 

    # Conversation History
    MAX_HISTORY: int = Field(default=20, ge=1, le=100)

    # Retry Configuration
    RETRY_ATTEMPTS: int = Field(default=3, ge=1, le=10)
    RETRY_DELAY: float = Field(default=0.2, gt=0, le=60)

    # Logging
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    LOG_FILE: str | None = None
    LOG_JSON: bool = False

    @model_validator(mode="after")
    def validate_keys(self) -> "Config":
        """Validate that required API keys are present for the selected provider."""
        if self.LLM_PROVIDER == "openai" and not (self.API_KEY or self.OPENAI_API_KEY):
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.LLM_PROVIDER == "groq" and not (self.GROQ_API_KEY or self.API_KEY):
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        if self.LLM_PROVIDER == "gemini" and not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        return self


# Lazy-loaded config instance
_config_instance: Config | None = None


def get_config() -> Config:
    """Get the global config instance (lazy-loaded with dev fallback)."""
    global _config_instance
    if _config_instance is None:
        try:
            _config_instance = Config()
        except ValidationError as e:
            if os.getenv("ENV") == "production":
                raise
            # Dev fallback: use mock provider with safe defaults
            _config_instance = Config(LLM_PROVIDER="mock")
    return _config_instance


def reload_config() -> Config:
    """Force reload config (useful for testing)."""
    global _config_instance
    _config_instance = None
    return get_config()