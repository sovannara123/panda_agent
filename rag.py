import chromadb
from datetime import datetime
from openai import OpenAI
from config import get_config
from logger import log_event
from document_parser import DocumentParser, DocumentParseError
from fastembed import TextEmbedding
from fastapi import FastAPI, HTTPException, UploadFile, File
import tempfile
import shutil
from pathlib import Path
from pydantic import BaseModel


# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_base")

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


def ingest_document(file_path: str, source_name: str = "") -> dict:
    """Parse a PDF document and ingest it into the vector database."""
    
    # Step 1: Parse the document
    parsed = DocumentParser.parse_pdf(file_path)
    metadata = parsed["metadata"]
    pages = parsed["pages"]
    
    # Use filename as source if not provided
    source = source_name or metadata["filename"]
    
    # Step 2: Chunk and embed each page
    all_chunks = []
    all_embeddings = []
    all_metadatas = []
    all_ids = []
    
    upload_timestamp = datetime.utcnow().isoformat()
    
    for page in pages:
        page_text = page["text"]
        page_number = page["page_number"]
        
        # Chunk this page
        chunks = chunk_text(page_text)
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{source}_page{page_number}_chunk{chunk_idx}"
            
            all_chunks.append(chunk)
            all_embeddings.append(get_embedding(chunk))
            all_metadatas.append({
                "source": source,
                "page_number": page_number,
                "chunk_index": chunk_idx,
                "uploaded_at": upload_timestamp,
                "total_pages": metadata["total_pages"]
            })
            all_ids.append(chunk_id)
    
    # Step 3: Add to ChromaDB
    if all_chunks:
        collection.add(
            documents=all_chunks,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )
    
    result = {
        "source": source,
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
    
    all_chunks = []
    all_embeddings = []
    all_metadatas = []
    all_ids = []
    
    upload_timestamp = datetime.utcnow().isoformat()
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_id = f"{source_name}_chunk{chunk_idx}"
        
        all_chunks.append(chunk)
        all_embeddings.append(get_embedding(chunk))
        all_metadatas.append({
            "source": source_name,
            "chunk_index": chunk_idx,
            "uploaded_at": upload_timestamp,
            "total_pages": 1
        })
        all_ids.append(chunk_id)
    
    if all_chunks:
        collection.add(
            documents=all_chunks,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )
    
    result = {
        "source": source_name,
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
    
    query_embedding = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    documents = results.get("documents") or []
    if not documents or not documents[0]:
        return {
            "found": False,
            "context": "No relevant information found in the knowledge base.",
            "sources": []
        }

    # Chroma may return metadata entries as None when no metadata exists.
    metadata_list = results.get("metadatas") or []
    top_metadata = metadata_list[0] if metadata_list and metadata_list[0] is not None else []

    # Ensure top_metadata is a list (not None) for type checker
    if top_metadata is None:
        top_metadata = []

    # Build context with source tracking
    context_parts = []
    sources = []

    for i, doc in enumerate(documents[0]):
        meta = top_metadata[i] if i < len(top_metadata) and top_metadata[i] is not None else {}
        source = meta.get("source", "unknown")
        page = meta.get("page_number", "?")

        context_parts.append(
            f"[Source: {source}, Page {page}]\n{doc}"
        )

        # Track unique sources
        source_info = {
            "source": source,
            "page": page
        }
        if source_info not in sources:
            sources.append(source_info)
    
    return {
        "found": True,
        "context": "\n\n---\n\n".join(context_parts),
        "sources": sources
    }


def list_documents() -> list[dict]:
    """List all unique documents in the knowledge base."""
    
    all_data = collection.get(include=["metadatas"])
    
    if not all_data["metadatas"]:
        return []
    
    # Group by source
    docs = {}
    for meta in all_data["metadatas"]:
        source = meta.get("source")
        if source and source not in docs:
            docs[source] = {
                "source": source,
                "total_pages": meta.get("total_pages", 0),
                "uploaded_at": meta.get("uploaded_at", "unknown"),
                "chunk_count": 0
            }
        if source:
            docs[source]["chunk_count"] += 1
    
    return list(docs.values())


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