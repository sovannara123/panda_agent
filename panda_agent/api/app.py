from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, Request, status
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
import time
import os

from panda_agent.agent.async_agent import AsyncAgent
from panda_agent.core.logger import log_event
from panda_agent.core.config import get_config
from panda_agent.rag.pipeline import ingest_document
from panda_agent.schemas.schemas import ChatRequest, ChatResponse, HealthResponse
from panda_agent.core.observability import setup_telemetry, tracer, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# --- Security Dependencies & Rate Limiting ---

_request_timestamps: dict[str, list[float]] = {}


async def check_rate_limit(request: Request):
    """Enforce per-IP request rate limiting."""
    config = get_config()
    client_ip = request.client.host if request.client else "127.0.0.1"
    now = time.time()
    window = 60.0
    max_requests = config.RATE_LIMIT_PER_MINUTE

    timestamps = _request_timestamps.get(client_ip, [])
    timestamps = [t for t in timestamps if now - t < window]

    if len(timestamps) >= max_requests:
        log_event("rate_limit_exceeded", {"client_ip": client_ip, "limit": max_requests})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )

    timestamps.append(now)
    _request_timestamps[client_ip] = timestamps


async def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization")
):
    """Validate incoming client API key against server configuration."""
    config = get_config()
    if not config.REQUIRE_API_KEY:
        return True

    expected_key = config.API_KEY
    if not expected_key:
        return True

    token = x_api_key
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1].strip()

    if token != expected_key:
        log_event("auth_failure", {"provided_key": bool(token)})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    return True


def sanitize_session_id(session_id: str | None) -> str:
    """Sanitize and validate session ID to prevent injection attacks."""
    if not session_id:
        return str(uuid.uuid4())
    clean_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
    return clean_id if clean_id else str(uuid.uuid4())


# Global agent instance removed for thread safety
# Instead, an agent is instantiated per request

@asynccontextmanager # async context manager for lifespan events 
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    print("🚀 Starting up Panda Agent API...")
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

# Instrument the FastAPI app with OpenTelemetry
setup_telemetry(app)

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


@app.get("/metrics", dependencies=[Depends(verify_api_key)])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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

@app.post("/upload", response_model=UploadResponse, dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def upload_pdf(file: UploadFile = File(...)):   
    """Upload a PDF file, extract text, and ingest into ChromaDB."""
    
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Sanitize filename against Path Traversal (e.g. "../../etc/passwd" -> "passwd")
    clean_filename = Path(file.filename).name
    
    # Enforce maximum upload size (e.g. 10MB)
    max_bytes = get_config().MAX_UPLOAD_SIZE_MB * 1024 * 1024
    uploaded_bytes = 0
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
        while chunk := await file.read(64 * 1024):
            uploaded_bytes += len(chunk)
            if uploaded_bytes > max_bytes:
                tmp.close()
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                log_event("upload_rejected_oversized", {"filename": clean_filename, "size": uploaded_bytes})
                raise HTTPException(
                    status_code=413,
                    detail=f"Payload Too Large. Maximum upload size is {get_config().MAX_UPLOAD_SIZE_MB}MB."
                )
            tmp.write(chunk)
    
    try:
        # Ingest document into ChromaDB (offload to thread pool to avoid blocking event loop)
        result = await asyncio.to_thread(ingest_document, tmp_path, clean_filename)
        
        log_event("pdf_uploaded", {
            "filename": clean_filename,
            "total_pages": result["total_pages"],
            "total_chunks": result["total_chunks"]
        })
        
        return UploadResponse(
            source=result["source"],
            filename=clean_filename,
            total_pages=result["total_pages"],
            extracted_pages=result["extracted_pages"],
            total_chunks=result["total_chunks"],
            uploaded_at=result["uploaded_at"]
        )
        
    except HTTPException:
        raise
    except Exception as error:
        log_event("pdf_upload_error", {"filename": clean_filename, "error": str(error)})
        raise HTTPException(status_code=500, detail="Failed to process document")
    
    finally:
        # Clean up temp file
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


@app.get("/documents", dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def list_documents():
    """List all ingested documents in the knowledge base."""
    from panda_agent.rag.pipeline import list_documents as rag_list_documents
    return {"documents": rag_list_documents()}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint."""

    # Sanitize and validate session ID
    session_id = sanitize_session_id(request.session_id)
    
    with tracer.start_as_current_span("agent.reasoning_loop") as root_span:
        root_span.set_attribute("session_id", session_id)
        root_span.set_attribute("user_input.length", len(request.message))
        
        # Initialize a new agent isolated to this session
        agent = AsyncAgent(session_id=session_id)
        
        try:
            response_text = await agent.respond_async(
                user_input=request.message,
                session_id=session_id
            )
            
            root_span.set_attribute("response.length", len(response_text))
            return ChatResponse(
                response=response_text,
                session_id=session_id
            )
            
        except HTTPException:
            raise
        except Exception as error:
            log_event("api_error", {"error": str(error), "message": request.message})
            root_span.record_exception(error)
            raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/chat/stream", dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint."""
    
    session_id = sanitize_session_id(request.session_id)
    agent = AsyncAgent(session_id=session_id)
    
    async def generate():
        with tracer.start_as_current_span("agent.reasoning_loop") as root_span:
            root_span.set_attribute("session_id", session_id)
            root_span.set_attribute("user_input.length", len(request.message))
            
            try:
                async for chunk in agent.respond_stream_async(
                    user_input=request.message,
                    session_id=session_id
                ):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
            except Exception as error:
                log_event("api_error", {"error": str(error), "message": request.message})
                root_span.record_exception(error)
                yield f"data: {json.dumps({'error': 'An internal server error occurred'})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")



