import chromadb
from datetime import datetime
from openai import OpenAI
from config import get_config
from logger import log_event
from document_parser import DocumentParser
from sentence_transformers import SentenceTransformer


# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_base")

# Lazy-loaded clients
_openai_client: OpenAI | None = None
_local_model: SentenceTransformer | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        config = get_config()
        api_key = config.OPENAI_API_KEY or config.API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _get_local_model() -> SentenceTransformer:
    global _local_model
    if _local_model is None:
        config = get_config()
        _local_model = SentenceTransformer(config.EMBEDDING_MODEL)
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
        return model.encode(text).tolist()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
            
    return chunks


def ingest_document(file_path: str, source_name: str = None) -> dict:
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


def search_knowledge_base(query: str, k: int = 3) -> dict:
    """Search the vector database for relevant context.
    
    Returns structured result with sources for citations.
    """
    
    query_embedding = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    if not results["documents"] or not results["documents"][0]:
        return {
            "found": False,
            "context": "No relevant information found in the knowledge base.",
            "sources": []
        }
    
    # Build context with source tracking
    context_parts = []
    sources = []
    
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
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