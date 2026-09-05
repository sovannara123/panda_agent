from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uuid

from async_agent import AsyncAgent
from logger import log_event

# Global agent instance (initialized once on startup)
agent_instance = None


@asynccontextmanager # async context manager for lifespan events 
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    global agent_instance
    print("🚀 Starting up Panda Agent API...")
    agent_instance = AsyncAgent() # 
    log_event("api_startup", {"status": "ready"})
    yield
    print("🛑 Shutting down Panda Agent API...")
    log_event("api_shutdown", {"status": "stopped"})

# this create fast api application instance with metadata and lifespan context manager 
app = FastAPI(
    title="Panda Agent API",
    description="Production-ready AI Agent API with tools and RAG",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


# --- Endpoints ---   

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "healthy", "agent_ready": agent_instance is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint."""

    # check whether agent exist if not return 503 service unavailable error
    if agent_instance is None: 
        raise HTTPException(status_code=503, detail="Agent is not initialized")
    
    # Generate a session ID if not provided
    # Use the user's session ID if they gave one; otherwise create a new one.
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # This is where FastAPI hands the request to your AI brain.
        # Wait for the asynchronous agent operation to finish.
        response_text = await agent_instance.respond_async(
            user_input=request.message,
            session_id=session_id
        )

        # now after API sends the answer back.
        return ChatResponse(
            response=response_text,
            session_id=session_id
        )
        
    except Exception as error:
        log_event("api_error", {"error": str(error), "message": request.message})
        raise HTTPException(status_code=500, detail="Internal server error")