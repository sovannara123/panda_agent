import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from config import get_config
from logger import log_event

# Initialize ChromaDB (persistent to disk in ./chromadb)
chroma_client = chromadb.PersistentClient(path="./chromadb")

collection = chroma_client.get_or_create_collection(
    name="KnowledgeBase",
    metadata={"description": "Knowledge base for RAG system"}
)

# Lazy-loaded embedding model (local)
_local_embedding_model: SentenceTransformer | None = None
_openai_client: OpenAI | None = None


def _get_local_model() -> SentenceTransformer:
    global _local_embedding_model
    if _local_embedding_model is None:
        config = get_config()
        _local_embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _local_embedding_model


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        config = get_config()
        api_key = config.OPENAI_API_KEY or config.API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_embedding(text: str) -> list[float]:
    """Generate embedding for piece of text using configured provider."""
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
    """Chunk text into smaller pieces with overlap"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def ingest_documents(documents: list[str], source: str = "manual"):
    """Ingest and index documents into the vector database"""

    all_chunks = []
    all_embeddings = []
    all_metadatas = []
    all_ids = []

    for doc_idx, doc in enumerate(documents):
        chunks = chunk_text(doc)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{source}_doc{doc_idx}_chunk{chunk_idx}"

            all_chunks.append(chunk)
            all_embeddings.append(get_embedding(chunk))
            all_metadatas.append({"source": source, "chunk_index": chunk_idx})
            all_ids.append(chunk_id)

            log_event("rag_ingest", {
                "chunk_id": chunk_id,
                "source": source,
                "chunk_length": len(chunk)
            })

    # Add to ChromaDB
    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
        ids=all_ids
    )

    log_event("rag_ingest_complete", {
        "total_chunks": len(all_chunks),
        "source": source
    })


def search_knowledge_base(query: str, k: int = 3) -> str:
    """Search the vector database for relevant context."""

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    if not results["documents"] or not results["documents"][0]:
        return "No relevant information found in the knowledge base."

    context_parts = []
    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i].get("source", "unknown") # type: ignore
        context_parts.append(f"[Source: {source}]\n{doc}")

    return "\n\n---\n\n".join(context_parts)