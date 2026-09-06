from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uuid
import json

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
    
    # Type narrowing: agent_instance is guaranteed to be AsyncAgent after the check
    agent: AsyncAgent = agent_instance  # type: ignore[assignment]
    
    # Generate a session ID if not provided
    # Use the user's session ID if they gave one; otherwise create a new one.
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # This is where FastAPI hands the request to your AI brain.
        # Wait for the asynchronous agent operation to finish.
        response_text = await agent.respond_async(
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


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint."""
    
    if agent_instance is None:
        raise HTTPException(status_code=503, detail="Agent is not initialized")
    
    # Type narrowing: agent_instance is guaranteed to be AsyncAgent after the check
    agent: AsyncAgent = agent_instance  # type: ignore[assignment]
    
    session_id = request.session_id or str(uuid.uuid4())
    
    async def generate():
        try:
            async for chunk in agent.respond_stream_async(
                user_input=request.message,
                session_id=session_id
            ):
                # Format as SSE
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
        except Exception as error:
            log_event("api_error", {"error": str(error), "message": request.message})
            yield f"data: {json.dumps({'error': str(error)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")