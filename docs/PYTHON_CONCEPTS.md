# Python Concepts Used in Panda Agent

This document catalogs the Python language features, design patterns, and standard library modules used throughout the project.

---

## Core Language Features

### Type Hints & Annotations (PEP 484, 585, 604)
```python
# Union types (Python 3.10+)
session_id: str | None = None
limit: int | None = None

# Generic types
messages: list[dict]
tool_calls: list[dict] | None
AsyncGenerator[dict, None]

# TypedDict for structured dicts (schemas.py)
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(default=None)
```

### Dataclasses (Python 3.7+)
```python
# context.py
@dataclass
class RequestContext:
    session_id: str
    request_id: str
    
    @classmethod
    def new(cls, session_id: str | None = None) -> "RequestContext":
        return cls(
            session_id=session_id or str(uuid.uuid4()),
            request_id=str(uuid.uuid4())
        )
```

### Abstract Base Classes (ABC)
```python
# llm_adapters.py
class LLMClient(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def generate(self, prompt: str) -> dict: ...
    
    @abstractmethod
    async def generate_async(self, prompt: str) -> dict: ...
```

### Pattern Matching (match/case) - Not Used
*Project uses if/elif chains instead (compatible with Python 3.10+)*

---

## Modern Python Features (3.10+)

### Structural Pattern Matching Alternative
```python
# tools.py - uses if/elif for tool planning
if "test failure" in msg or "simulate failure" in msg:
    return {"tool": "test_failure", "arguments": {}}

if any(word in msg for word in PRICE_KEYWORDS):
    ...
```

### Walrus Operator (:=)
```python
# Not explicitly used, but available in Python 3.8+
```

### Exception Groups (Python 3.11+)
```python
# Not used - single exception handling with retry.py
```

---

## Standard Library Modules

### Asyncio - Asynchronous Programming
```python
# async_agent.py
import asyncio

async def _handle_tool_calls_async(self, ...):
    # Concurrent execution
    tool_results = await asyncio.gather(*[
        self._execute_single_tool_async(tool_call, ctx)
        for tool_call in tool_calls
    ])

# Thread pool for blocking I/O
loop = asyncio.get_event_loop()
tool_result = await loop.run_in_executor(
    None, execute_tool, {"tool": tool_name, "arguments": arguments}
)
```

### Async Generators
```python
# llm_adapters.py
async def generate_stream_async(self, prompt: str) -> AsyncGenerator[dict, None]:
    async for chunk in stream:
        yield {"type": "content", "content": chunk}

# Consumed with async for
async for chunk in self.llm.generate_with_tools_stream_async(...):
    ...
```

### Context Managers
```python
# storage.py - Database connections
@contextmanager
def _get_connection(self):
    conn = sqlite3.connect(self.db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# Usage
with self._get_connection() as conn:
    conn.execute(...)
```

### Pathlib (Path Objects)
```python
# storage.py
from pathlib import Path

self.db_path = Path(db_path)
self.messages_path.write_text(json.dumps(messages), encoding="utf-8")
```

### UUID
```python
# context.py
import uuid

session_id=session_id or str(uuid.uuid4())
request_id=str(uuid.uuid4())
```

### Datetime & Time
```python
# retry.py
import time
time.sleep(delay_seconds * (2 ** (attempt - 1)))

# SQLite schema uses TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

### JSON
```python
# Multiple files
import json
json.dumps(data, indent=2, ensure_ascii=False)
json.loads(text)
```

### Logging
```python
# logger.py
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
```

### Regex (re)
```python
# tools.py
import re

order_match = re.search(r'\b[aA]\d{3}\b', message)
re.search(rf'\b{re.escape(product)}\b', msg)
```

### Typing Module
```python
from typing import AsyncGenerator, Generator, Optional, TYPE_CHECKING, Any
from typing import TYPE_CHECKING  # For forward references
```

---

## Design Patterns

### 1. Strategy Pattern - LLM Providers
```python
# llm_adapters.py
class LLMClient(ABC): ...

class OpenAIClient(LLMClient): ...
class GroqClient(LLMClient): ...
class OllamaClient(LLMClient): ...
class MockLLMClient(LLMClient): ...

def create_llm_client(provider: str | None = None) -> LLMClient:
    provider = (provider or get_config().LLM_PROVIDER).lower()
    if provider == "openai": return OpenAIClient(...)
    if provider == "groq": return GroqClient(...)
    ...
```

### 2. Factory Pattern - Config & Client Creation
```python
# config.py - Singleton with lazy initialization
_config_instance: Config | None = None

def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()  # or fallback
    return _config_instance                   
```

### 3. Template Method Pattern - Response Flows
```python
# agent.py
def respond(self, user_input: str, session_id: str | None = None) -> str:
    if hasattr(self.llm, 'generate_with_tools') and not isinstance(self.llm, MockLLMClient):
        return self.respond_with_function_calling(user_input, session_id)
    else:
        return self._respond_legacy(user_input, session_id)
```

### 4. Decorator Pattern - Retry Logic
```python
# retry.py
def retry_with_backoff(func, max_attempts=3, delay_seconds=0.1, exceptions=(Exception,)):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except exceptions as error:
            if attempt == max_attempts:
                break
            time.sleep(delay_seconds * (2 ** (attempt - 1)))
    raise RetryError(...) from last_error

# Usage in agent.py
llm_output = retry_with_backoff(
    lambda: self.llm.generate(prompt),
    max_attempts=get_config().RETRY_ATTEMPTS,
    delay_seconds=get_config().RETRY_DELAY,
)
```

### 5. Observer Pattern - Structured Logging
```python
# logger.py
def log_event(event_type: str, data: dict, context: RequestContext | None = None):
    log.info(json.dumps(_get_base_data(event_type, data, context)))

# Correlation IDs injected via filter
class CorrelationFilter(logging.Filter):
    def filter(self, record):
        record.session_id = getattr(record, "session_id", "-")
        record.request_id = getattr(record, "request_id", "-")
        return True
```

### 6. Dependency Injection
```python
# agent.py
def __init__(self, storage=None, llm_client=None, session_id: str | None = None):
    self.storage = storage or SQLiteStorage()
    self.llm = llm_client or create_llm_client()
    # Easy to inject MockLLMClient for testing
```

### 7. Adapter Pattern - Tool Schema Conversion
```python
# tool_formatter.py
def convert_to_openai_tools(schemas: dict) -> list:
    for tool_name, schema in schemas.items():
        openai_tool = {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"]
            }
        }
```

---

## Pydantic (v2) - Data Validation

### BaseModel with Field Validation
```python
# schemas.py
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(default=None)
    
    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v: raise ValueError("Message cannot be empty")
        if any(p in v.lower() for p in ["<script", "javascript:"]):
            raise ValueError("Invalid message content")
        return v
```

### Settings Management (pydantic-settings)
```python
# config.py
class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    
    LLM_PROVIDER: str = Field(default="mock", pattern="^(mock|openai|groq|ollama|gemini)$")
    MESSAGE_LIMIT_FREE: int = Field(default=10, ge=1, le=10000)
    
    @model_validator(mode="after")
    def validate_keys(self) -> "Config":
        if self.LLM_PROVIDER == "openai" and not (self.API_KEY or self.OPENAI_API_KEY):
            raise ValueError("OPENAI_API_KEY is required...")
        return self
```

---

## Concurrency Concepts

### Thread Pool Executor (for Blocking I/O)
```python
# async_agent.py
loop = asyncio.get_event_loop()
tool_result = await loop.run_in_executor(
    None,  # default ThreadPoolExecutor
    execute_tool,
    {"tool": tool_name, "arguments": arguments}
)
```

### AsyncIO Gather (Parallel Execution)
```python
# async_agent.py
tool_results = await asyncio.gather(*[
    self._execute_single_tool_async(tool_call, ctx)
    for tool_call in tool_calls
])
```

---

## Error Handling Patterns

### Custom Exception Hierarchy
```python
# llm_adapters.py
class LLMError(Exception): pass
class RetryError(LLMError): pass

# tools.py
class UnknownToolError(Exception):
    def __init__(self, tool_name):
        self.tool_name = tool_name
        super().__init__(f"Unknown tool: {tool_name}")

class ToolArgumentError(Exception): pass
```

### Try/Except with Specific Exceptions
```python
# agent.py
try:
    llm_output = self.generate_llm_response(prompt)
except RetryError as error:
    response = get_fallback_response("llm_unavailable", details=str(error))
except Exception as error:
    response = get_fallback_response("llm_unavailable", details=str(error))
```

### Exception Chaining
```python
# retry.py
raise RetryError(f"Operation failed after {max_attempts} attempts.") from last_error
```

---

## Testing Patterns

### Pytest Fixtures
```python
# test_agent.py
@pytest.fixture
def agent(tmp_path):
    agent = Agent(storage=MemoryStorage(
        str(tmp_path / "memory.json"),
        str(tmp_path / "meta.json")
    ))
    agent.llm = MockLLMClient()
    return agent
```

### Monkeypatching
```python
def test_custom_limit_from_config(self, agent, monkeypatch):
    monkeypatch.setenv("MESSAGE_LIMIT_FREE", "3")
    reload_config()
    ...
```

### Caplog for Log Testing
```python
def test_fallback_logged_on_usage_limit(self, agent, caplog, monkeypatch):
    monkeypatch.setattr(log, "propagate", True)
    caplog.set_level(logging.INFO, logger="agent")
    agent.respond("hello")
    messages = [r.getMessage() for r in caplog.records if '"event": "fallback"' in r.getMessage()]
```

---

## SQL/SQLite Patterns

### Parameterized Queries (SQL Injection Prevention)
```python
# storage.py
conn.execute(
    "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
    (sid, msg["role"], msg["content"])
)

conn.execute("SELECT ... WHERE session_id = ?", (session_id,))
```

### Row Factory for Dict-like Results
```python
conn.row_factory = sqlite3.Row
cursor = conn.execute(query, params)
return [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]
```

### Atomic Transactions
```python
@contextmanager
def _get_connection(self):
    conn = sqlite3.connect(...)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
```

---

## Generator Patterns

### Synchronous Generators (Streaming)
```python
# llm_adapters.py
def generate_stream(self, prompt: str):
    stream = self.client.chat.completions.create(..., stream=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield {"type": "content", "content": chunk.choices[0].delta.content}
    yield {"type": "done", "full_content": "".join(collected_content)}
```

### Consumer Side
```python
# main.py
for token in agent.respond_stream_simple(user_input, session_id=session_id):
    print(token, end="", flush=True)
```

---

## Configuration Management

### Environment Variables with Defaults
```python
# config.py
class Config(BaseSettings):
    LLM_PROVIDER: str = Field(default="mock", ...)
    MODEL_NAME: str = "gpt-4o-mini"
    API_KEY: str = ""
    GROQ_API_KEY: str | None = None
    
    # Validation at runtime
    @model_validator(mode="after")
    def validate_keys(self) -> "Config":
        if self.LLM_PROVIDER == "openai" and not (self.API_KEY or self.OPENAI_API_KEY):
            raise ValueError("OPENAI_API_KEY is required...")
        return self
```

### Development Fallback
```python
def get_config() -> Config:
    try:
        _config_instance = Config()
    except ValidationError as e:
        if os.getenv("ENV") == "production":
            raise
        _config_instance = Config(LLM_PROVIDER="mock")  # Dev fallback
    return _config_instance
```

---

## Summary: Concepts by Category

| Category | Concepts Used |
|----------|---------------|
| **Type System** | Type hints, Union (`|`), Optional, Generics, TypedDict, Pydantic models |
| **OOP** | ABC, inheritance, polymorphism, composition, dependency injection |
| **Async** | async/await, AsyncGenerator, asyncio.gather, run_in_executor, context managers |
| **Patterns** | Strategy, Factory, Template Method, Decorator, Observer, Adapter |
| **Data** | Pathlib, JSON, SQLite, UUID, regex, datetime |
| **Validation** | Pydantic v2 (Field, field_validator, model_validator, BaseSettings) |
| **Error Handling** | Custom exceptions, exception chaining, specific catch, retry with backoff |
| **Testing** | Fixtures, monkeypatch, caplog, parameterized tests |
| **Logging** | Structured JSON, correlation IDs, rotating file handler, filters |

---

## Python Version Requirements

**Minimum: Python 3.10** (for `str | None` union syntax)

**Recommended: Python 3.11+** (better performance, exception groups, TOML support)

**Used: Python 3.13** (per pyrightconfig.json)