import uuid
from typing import Any
import chromadb
from datetime import datetime, timezone
import datetime
import tempfile
import pathlib
import time
import os
from openai import OpenAI
from panda_agent.core.config import get_config
from panda_agent.core.logger import log_event
from panda_agent.core.observability import tracer, RAG_LATENCY
from panda_agent.rag.document_parser import DocumentParser, DocumentParseError
from fastembed import TextEmbedding
from fastapi import FastAPI, HTTPException, UploadFile, File
import shutil
from pathlib import Path
from pydantic import BaseModel


from panda_agent.rag.vector_store import get_vector_store

# Lazy-loaded clients
_openai_client: OpenAI | None = None
_local_model: TextEmbedding | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        config = get_config()
        api_key = config.OPENAI_API_KEY or config.API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _get_local_model() -> TextEmbedding:
    global _local_model
    if _local_model is None:
        config = get_config()
        _local_model = TextEmbedding(model_name=config.EMBEDDING_MODEL)
    return _local_model


def get_embedding(text: str) -> list[float]:
    """Generate embedding for a piece of text."""
    config = get_config()

    if config.EMBEDDING_PROVIDER == "openai":
        client = _get_openai_client()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    else:
        model = _get_local_model()
        return list(model.embed(text))[0].tolist()


def get_embeddings_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Generate embeddings for multiple texts in batches.
    
    OpenAI supports up to 2048 texts per call. We batch at 100 to balance
    throughput and memory usage. This turns 3,000 API calls into 30.
    """
    config = get_config()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        if config.EMBEDDING_PROVIDER == "openai":
            client = _get_openai_client()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])
        else:
            model = _get_local_model()
            all_embeddings.extend([e.tolist() for e in model.embed(batch)])

    return all_embeddings


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
            
    return chunks


class UploadResponse(BaseModel):
    source: str
    total_pages: int
    extracted_pages: int
    total_chunks: int
    message: str


class DocumentListResponse(BaseModel):
    documents: list[dict]
    count: int


app = FastAPI(title="Panda Agent RAG API", version="1.0.0")


def _find_page_for_position(char_pos: int, page_boundaries: list[tuple[int, int, int]]) -> int:
    """Find which page a character position falls in."""
    for start, end, page_num in page_boundaries:
        if start <= char_pos < end:
            return page_num
    # Default to last page if position is at the very end
    return page_boundaries[-1][2] if page_boundaries else 1


def ingest_document(file_path: str, source_name: str = "") -> dict:
    """Parse a PDF document and ingest it into the vector database.
    
    Fixes applied:
    - Cross-page chunking: pages are concatenated before chunking so overlap
      spans page boundaries (no more lost answers at page breaks).
    - UUID chunk IDs: each upload gets a unique ID to prevent collisions
      when two files share the same name.
    - Batched embeddings: chunks are embedded in batches of 100 instead of
      one API call per chunk (3,000 calls → 30 calls).
    - Batched upserts: ChromaDB writes happen in batches of 200 to limit
      peak memory usage.
    """
    
    # Step 1: Parse the document
    parsed = DocumentParser.parse_pdf(file_path)
    metadata = parsed["metadata"]
    pages = parsed["pages"]
    
    # Enforce maximum page limit guard (max 50 pages)
    MAX_PAGES = 50
    if len(pages) > MAX_PAGES:
        pages = pages[:MAX_PAGES]
        metadata["extracted_pages"] = MAX_PAGES
    
    # Use filename as source if not provided
    source = source_name or metadata["filename"]
    
    # Unique ID per upload — prevents chunk ID collisions between
    # different files that share the same filename.
    upload_id = uuid.uuid4().hex[:8]
    
    upload_timestamp = datetime.now(timezone.utc).isoformat()
    
    # Step 2: Concatenate all pages into a single text with boundary tracking.
    # This ensures the overlap window in chunk_text() naturally crosses page
    # boundaries, so answers spanning two pages are never split.
    full_text = ""
    page_boundaries: list[tuple[int, int, int]] = []  # (char_start, char_end, page_number)
    
    for page in pages:
        start = len(full_text)
        full_text += page["text"] + " "
        page_boundaries.append((start, len(full_text), page["page_number"]))
    
    # Step 3: Chunk the full concatenated text
    chunks = chunk_text(full_text)
    
    # Build chunk metadata with page number mapping
    all_chunks: list[str] = []
    all_metadatas: list[dict[str, Any]] = []
    all_ids: list[str] = []
    
    # Track where each chunk starts in full_text for page mapping
    search_start = 0
    for chunk_idx, chunk in enumerate(chunks):
        # Find this chunk's position in the full text to determine its page
        chunk_pos = full_text.find(chunk[:100], search_start)  # use first 100 chars for faster search
        if chunk_pos == -1:
            chunk_pos = search_start  # fallback
        search_start = max(search_start, chunk_pos + 1)
        
        page_number = _find_page_for_position(chunk_pos, page_boundaries)
        
        chunk_id = f"{upload_id}_{source}_page{page_number}_chunk{chunk_idx}"
        
        all_chunks.append(chunk)
        all_metadatas.append({
            "source": source,
            "upload_id": upload_id,
            "page_number": page_number,
            "chunk_index": chunk_idx,
            "uploaded_at": upload_timestamp,
            "total_pages": metadata["total_pages"]
        })
        all_ids.append(chunk_id)
    
    # Step 4: Batch embed all chunks (100 per API call instead of 1)
    all_embeddings = get_embeddings_batch(all_chunks) if all_chunks else []
    
    # Step 5: Upsert to vector store adapter
    vector_store = get_vector_store()
    vector_store.add_chunks(
        ids=all_ids,
        chunks=all_chunks,
        embeddings=all_embeddings,  # type: ignore[arg-type]
        metadatas=all_metadatas
    )
    
    result = {
        "source": source,
        "upload_id": upload_id,
        "total_pages": metadata["total_pages"],
        "extracted_pages": metadata["extracted_pages"],
        "total_chunks": len(all_chunks),
        "uploaded_at": upload_timestamp
    }
    
    log_event("document_ingested", result)
    
    return result


def ingest_text(text: str, source_name: str) -> dict:
    """Ingest raw text directly into the vector database (no PDF parsing)."""
    
    chunks = chunk_text(text)
    
    # Unique ID per upload to prevent collisions
    upload_id = uuid.uuid4().hex[:8]
    
    all_chunks: list[str] = []
    all_metadatas: list[dict[str, Any]] = []
    all_ids: list[str] = []
    
    upload_timestamp = datetime.now(timezone.utc).isoformat()
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_id = f"{upload_id}_{source_name}_chunk{chunk_idx}"
        
        all_chunks.append(chunk)
        all_metadatas.append({
            "source": source_name,
            "upload_id": upload_id,
            "chunk_index": chunk_idx,
            "uploaded_at": upload_timestamp,
            "total_pages": 1
        })
        all_ids.append(chunk_id)
    
    # Batch embed all chunks
    all_embeddings = get_embeddings_batch(all_chunks) if all_chunks else []
    
    # Upsert to vector store adapter
    vector_store = get_vector_store()
    vector_store.add_chunks(
        ids=all_ids,
        chunks=all_chunks,
        embeddings=all_embeddings,  # type: ignore[arg-type]
        metadatas=all_metadatas
    )
    
    result = {
        "source": source_name,
        "upload_id": upload_id,
        "total_pages": 1,
        "extracted_pages": 1,
        "total_chunks": len(all_chunks),
        "uploaded_at": upload_timestamp
    }
    
    log_event("text_ingested", result)
    return result


def search_knowledge_base(query: str, k: int = 3) -> dict:
    """Search the vector database for relevant context.
    
    Returns structured result with sources for citations.
    """
    with tracer.start_as_current_span("rag.retrieval") as span:
        start_time = time.time()
        
        query_embedding = get_embedding(query)
        
        vector_store = get_vector_store()
        matches = vector_store.query(query_embedding=query_embedding, top_k=k)
        
        duration = time.time() - start_time
        RAG_LATENCY.observe(duration)
        span.set_attribute("rag.latency_seconds", duration)
        
        if not matches:
            span.set_attribute("rag.chunks_retrieved", 0)
            return {
                "found": False,
                "context": "No relevant information found in the knowledge base.",
                "sources": []
            }
        
        span.set_attribute("rag.chunks_retrieved", len(matches))

    # Build context with source tracking
    context_parts = []
    sources = []

    for item in matches:
        doc = item["chunk"]
        meta = item.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page_number", "?")

        context_parts.append(
            f"[Source: {source}, Page {page}]\n{doc}"
        )

        source_info = {
            "source": source,
            "page": page
        }
        if source_info not in sources:
            sources.append(source_info)
    
    raw_context = "\n\n---\n\n".join(context_parts)
    fenced_context = f"<retrieved_context>\n{raw_context}\n</retrieved_context>"

    return {
        "found": True,
        "context": fenced_context,
        "sources": sources
    }


def list_documents() -> list[dict]:
    """List all unique documents in the knowledge base."""
    vector_store = get_vector_store()
    return vector_store.list_documents()


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF document."""
    
    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Save uploaded file temporarily
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / file.filename
    
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest the document
        result = ingest_document(str(temp_path))
        
        return UploadResponse(
            source=result["source"],
            total_pages=result["total_pages"],
            extracted_pages=result["extracted_pages"],
            total_chunks=result["total_chunks"],
            message=f"Successfully ingested {result['total_chunks']} chunks from {result['total_pages']} pages."
        )
        
    except DocumentParseError as error:
        raise HTTPException(status_code=400, detail=str(error))
        
    except Exception as error:
        log_event("upload_error", {"error": str(error), "filename": file.filename})
        raise HTTPException(status_code=500, detail="Failed to process document")
        
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


@app.get("/documents", response_model=DocumentListResponse)
async def get_documents():
    """List all documents in the knowledge base."""
    docs = list_documents()
    return DocumentListResponse(
        documents=docs,
        count=len(docs)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)