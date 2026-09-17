from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import uuid
import json
import shutil
import tempfile
import os

from panda_agent.agent.async_agent import AsyncAgent
from panda_agent.core.logger import log_event
from panda_agent.core.config import get_config
from panda_agent.rag.pipeline import ingest_document
from panda_agent.schemas.schemas import ChatRequest, ChatResponse, HealthResponse

# Global agent instance removed for thread safety
# Instead, an agent is instantiated per request

@asynccontextmanager # async context manager for lifespan events 
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    print("🚀 Starting up Panda Agent API...")
    log_event("api_startup", {"status": "ready"})
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

# Enable CORS securely based on configuration
allowed_origins_str = get_config().ALLOWED_ORIGINS
# Parse the comma-separated string into a list, handling whitespace
allowed_origins_list = [origin.strip() for origin in allowed_origins_str.split(",")] if allowed_origins_str else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return RedirectResponse(url="/docs")


# --- Endpoints ---   

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "healthy", "agent_ready": True, "version": "1.0.0"}


class UploadResponse(BaseModel):
    source: str
    filename: str
    total_pages: int
    extracted_pages: int
    total_chunks: int
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: list[dict]
    count: int

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):   
    """Upload a PDF file, extract text, and ingest into ChromaDB."""
    
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file to temp location (stream to disk, not into RAM)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        # Ingest document into ChromaDB (offload to thread pool to avoid blocking event loop)
        result = await asyncio.to_thread(ingest_document, tmp_path, file.filename)
        
        log_event("pdf_uploaded", {
            "filename": file.filename,
            "total_pages": result["total_pages"],
            "total_chunks": result["total_chunks"]
        })
        
        return UploadResponse(
            source=result["source"],
            filename=file.filename,
            total_pages=result["total_pages"],
            extracted_pages=result["extracted_pages"],
            total_chunks=result["total_chunks"],
            uploaded_at=result["uploaded_at"]
        )
        
    except Exception as error:
        log_event("pdf_upload_error", {"filename": file.filename, "error": str(error)})
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(error)}")
    
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.get("/documents")
async def list_documents():
    """List all ingested documents in the knowledge base."""
    from panda_agent.rag.pipeline import list_documents as rag_list_documents
    return {"documents": rag_list_documents()}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint."""

    # Generate a session ID if not provided
    # Use the user's session ID if they gave one; otherwise create a new one.
    session_id = request.session_id or str(uuid.uuid4())
    
    # Initialize a new agent isolated to this session
    agent = AsyncAgent(session_id=session_id)
    
    try:
        # This is where FastAPI hands the request to AI brain.
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
    
    session_id = request.session_id or str(uuid.uuid4())
    agent = AsyncAgent(session_id=session_id)
    
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



