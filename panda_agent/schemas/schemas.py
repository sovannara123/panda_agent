from pydantic import BaseModel, Field, field_validator
from typing import Optional
from uuid import UUID
"""
                    API
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
      REQUESTS               RESPONSES
          │                     │
          ↓                     ↓
    ┌────────────┐        ┌─────────────┐
    │ChatRequest │        │ChatResponse │
    │ToolRequest │        │ToolResponse │
    └──────┬─────┘        │Health       │
           │              │Error        │
           │              │Session      │
           ↓              └─────────────┘
      🛡️ VALIDATE
           │
       ┌───┴───┐
       ↓       ↓
     VALID   INVALID
       ↓       ↓
     Agent   ERROR ❌
"""

"""
One-sentence algorithm

Receive data → check required fields → clean data → validate formats → validate custom rules → accept valid data or reject invalid data.
"""

# goal: to validate user chatrquest
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=4000, description="User message") # to check the message must requred (the field mean that value have to be fill ) at least 1 charator and less than 4000 
    session_id: Optional[str] = Field(default=None, description="Optional session ID") 
    """
            # Optional [str] : tell python that this field can hold be None or String 
            # field (...) : attach pydantic speific validation and metadata in to this variable 
            # description="Optional session ID": Metadata explaining what the field is for.
            # Pydantic uses this description when generating OpenAPI docs (like Swagger UI) or JSON schema exports."""
    user_id: Optional[str] = Field(default=None, description="Optional user ID")

    @field_validator("message") # tell the pydantic to run this function specifically whenever message field updated 
    @classmethod
    def validate_message(cls, v: str) -> str:
        # Strip whitespace
        v = v.strip() # remove extra space 
        # check if the message is empty 
        if not v:
            raise ValueError("Message cannot be empty")
        # Basic injection prevention
        if any (pattern in v.lower() for pattern in ["<script", "javascript:", "onerror=", "onload="]):
            raise ValueError("Invalid message content")
        return v

    @field_validator("session_id")
    @classmethod 
    def validate_session_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            # Validate UUID format
            try:
                UUID(v)
            except ValueError:
                raise ValueError("Invalid session_id format")
        return v
# define structure of response sent back to client 
class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session ID")
    request_id: Optional[str] = Field(default=None, description="Request ID")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")


# response model for health check
class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    agent_ready: bool = Field(default=True, description="Agent readiness status")
    version: Optional[str] = Field(default="1.0.0", description="Agent version")
    timestamp: Optional[str] = Field(default=None, description="Current timestamp")