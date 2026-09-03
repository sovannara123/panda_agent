from pydantic import BaseModel, Field, field_validator
from typing import Optional
from uuid import UUID
import re
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
# define structure of respone send back to client 
class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session ID")
    request_id: str = Field(..., description="Request ID")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")

# return this information whether Ai service is working 
class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Agent version")
    timestamp: str = Field(..., description="Current timestamp")

# create standard format for error 
class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    details: Optional[dict] = Field(default=None, description="Optional error details")

#Store information about a conversation session 
class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    message_count: int
    created_at: str
    last_activity: str

#make the ai requesting a valid rool and valide argument
class ToolCallRequest(BaseModel):
    """Request model for tool calls."""
    tool: str = Field(..., pattern="^(get_product_price|check_order_status|get_weather|test_failure)$")
    arguments: dict = Field(..., description="Tool arguments")

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, v: dict, info) -> dict:
        tool = info.data.get("tool") if hasattr(info, "data") else ""
        if tool == "get_product_price":
            if "product_name" not in v or not v["product_name"]:
                raise ValueError("product_name is required")
        elif tool == "check_order_status":
            if "order_id" not in v or not v["order_id"]:
                raise ValueError("order_id is required")
            if not re.match(r"^A\d{3}$", v["order_id"]):
                raise ValueError("Invalid order_id format (must be A followed by 3 digits)")
        elif tool == "get_weather":
            if "city" not in v or not v["city"]:
                raise ValueError("city is required")
        return v

# define the result after the tool has executed 
class ToolCallResponse(BaseModel):
    """Response model for tool calls."""
    tool: str
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None