import math
from abc import ABC, abstractmethod
from typing import Any
import chromadb
from panda_agent.core.config import get_config
from panda_agent.core.logger import get_logger

logger = get_logger(__name__)


class VectorStoreAdapter(ABC):
    """Abstract Base Class for Vector Database Adapters."""

    @abstractmethod
    def add_chunks(
        self,
        ids: list[str],
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]]
    ) -> None:
        """Add text chunks and embeddings to the store."""
        pass

    @abstractmethod
    def query(self, query_embedding: list[float], top_k: int = 3) -> list[dict[str, Any]]:
        """Query vector store using vector similarity search."""
        pass

    @abstractmethod
    def list_documents(self) -> list[dict[str, Any]]:
        """List summary of ingested documents."""
        pass

    @abstractmethod
    def delete_document(self, source: str) -> None:
        """Delete all chunks associated with a specific document source."""
        pass


class ChromaDBAdapter(VectorStoreAdapter):
    """ChromaDB implementation for persistent local vector storage."""

    def __init__(self, collection_name: str = "knowledge_base"):
        config = get_config()
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(
        self,
        ids: list[str],
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]]
    ) -> None:
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas    # type: ignore[arg-type]
        )

    def query(self, query_embedding: list[float], top_k: int = 3) -> list[dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        matched: list[dict[str, Any]] = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas_raw = results.get("metadatas")
            metas = metas_raw[0] if (metas_raw and metas_raw[0] is not None) else []
            dists_raw = results.get("distances")
            dists = dists_raw[0] if (dists_raw and dists_raw[0] is not None) else []
            for i, doc in enumerate(docs):
                matched.append({
                    "chunk": doc,
                    "metadata": metas[i] if i < len(metas) and metas[i] is not None else {},
                    "distance": dists[i] if i < len(dists) and dists[i] is not None else 0.0
                })
        return matched

    def list_documents(self) -> list[dict[str, Any]]:
        records = self.collection.get()
        if not records or not records.get("metadatas"):
            return []

        doc_summary: dict[str, dict[str, Any]] = {}
        for meta in records["metadatas"]:
            if not meta:
                continue
            source = str(meta.get("source") or "unknown")
            if source not in doc_summary:
                doc_summary[source] = {
                    "source": source,
                    "total_pages": meta.get("total_pages", 1),
                    "uploaded_at": meta.get("uploaded_at"),
                    "chunk_count": 0
                }
            doc_summary[source]["chunk_count"] += 1

        return list(doc_summary.values())

    def delete_document(self, source: str) -> None:
        records = self.collection.get(where={"source": source})
        if records and records.get("ids"):
            self.collection.delete(ids=records["ids"])


class QdrantAdapter(VectorStoreAdapter):
    """Qdrant Enterprise Vector Store Adapter."""

    def __init__(self, collection_name: str = "knowledge_base"):
        config = get_config()
        self.collection_name = collection_name
        self.url = config.QDRANT_URL
        self.api_key = config.QDRANT_API_KEY

        try:
            from qdrant_client import QdrantClient  # type: ignore[import-not-found,import-untyped]
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
            logger.info(f"Initialized Qdrant client connected to {self.url}")
        except ImportError:
            logger.warning("qdrant-client package not installed. QdrantAdapter will run in mock mode.")
            self.client = None

    def add_chunks(
        self,
        ids: list[str],
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]]
    ) -> None:
        if not self.client:
            logger.warning("Qdrant mock: Skipping point insertion")
            return

        from qdrant_client.models import PointStruct  # type: ignore[import-not-found,import-untyped]
        points = [
            PointStruct(
                id=ids[i],
                vector=embeddings[i],
                payload={"chunk": chunks[i], **metadatas[i]}
            )
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, query_embedding: list[float], top_k: int = 3) -> list[dict[str, Any]]:
        if not self.client:
            return []

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        return [
            {
                "chunk": hit.payload.get("chunk", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "chunk"},
                "distance": hit.score
            }
            for hit in search_result
        ]

    def list_documents(self) -> list[dict[str, Any]]:
        if not self.client:
            return []
        # Basic scroll list implementation
        points, _ = self.client.scroll(collection_name=self.collection_name, limit=100)
        doc_summary: dict[str, dict[str, Any]] = {}
        for pt in points:
            meta = pt.payload or {}
            source = str(meta.get("source") or "unknown")
            if source not in doc_summary:
                doc_summary[source] = {
                    "source": source,
                    "total_pages": meta.get("total_pages", 1),
                    "uploaded_at": meta.get("uploaded_at"),
                    "chunk_count": 0
                }
            doc_summary[source]["chunk_count"] += 1
        return list(doc_summary.values())

    def delete_document(self, source: str) -> None:
        if not self.client:
            return
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore[import-not-found,import-untyped]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            )
        )


class PineconeAdapter(VectorStoreAdapter):
    """Pinecone Cloud Enterprise Vector Store Adapter."""

    def __init__(self, index_name: str | None = None):
        config = get_config()
        self.index_name = index_name or config.PINECONE_INDEX_NAME
        self.api_key = config.PINECONE_API_KEY

        try:
            from pinecone import Pinecone  # type: ignore[import-not-found,import-untyped]
            if not self.api_key:
                raise ValueError("PINECONE_API_KEY is required for PineconeAdapter")
            pc = Pinecone(api_key=self.api_key)
            self.index = pc.Index(self.index_name)
            logger.info(f"Initialized Pinecone index: {self.index_name}")
        except Exception as e:
            logger.warning(f"Pinecone client initialization failed ({e}). Operating in fallback mode.")
            self.index = None

    def add_chunks(
        self,
        ids: list[str],
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]]
    ) -> None:
        if not self.index:
            return
        vectors = [
            (ids[i], embeddings[i], {"chunk": chunks[i], **metadatas[i]})
            for i in range(len(ids))
        ]
        self.index.upsert(vectors=vectors)

    def query(self, query_embedding: list[float], top_k: int = 3) -> list[dict[str, Any]]:
        if not self.index:
            return []
        res = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        return [
            {
                "chunk": match.metadata.get("chunk", ""),
                "metadata": {k: v for k, v in match.metadata.items() if k != "chunk"},
                "distance": match.score
            }
            for match in res.matches
        ]

    def list_documents(self) -> list[dict[str, Any]]:
        return []

    def delete_document(self, source: str) -> None:
        if not self.index:
            return
        self.index.delete(filter={"source": {"$eq": source}})


class InMemoryVectorAdapter(VectorStoreAdapter):
    """In-memory cosine-similarity vector store adapter for isolated testing."""

    def __init__(self):
        self.store: list[dict[str, Any]] = []

    def add_chunks(
        self,
        ids: list[str],
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]]
    ) -> None:
        for i in range(len(ids)):
            self.store.append({
                "id": ids[i],
                "chunk": chunks[i],
                "embedding": embeddings[i],
                "metadata": metadatas[i]
            })

    def query(self, query_embedding: list[float], top_k: int = 3) -> list[dict[str, Any]]:
        if not self.store:
            return []

        def _cosine_sim(v1: list[float], v2: list[float]) -> float:
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

        scored = []
        for item in self.store:
            sim = _cosine_sim(query_embedding, item["embedding"])
            scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "chunk": item["chunk"],
                "metadata": item["metadata"],
                "distance": sim
            }
            for sim, item in scored[:top_k]
        ]

    def list_documents(self) -> list[dict[str, Any]]:
        doc_summary: dict[str, dict[str, Any]] = {}
        for item in self.store:
            meta = item["metadata"]
            source = meta.get("source", "unknown")
            if source not in doc_summary:
                doc_summary[source] = {
                    "source": source,
                    "total_pages": meta.get("total_pages", 1),
                    "uploaded_at": meta.get("uploaded_at"),
                    "chunk_count": 0
                }
            doc_summary[source]["chunk_count"] += 1
        return list(doc_summary.values())

    def delete_document(self, source: str) -> None:
        self.store = [item for item in self.store if item["metadata"].get("source") != source]


_vector_store_instance: VectorStoreAdapter | None = None


def get_vector_store() -> VectorStoreAdapter:
    """Factory function returning configured vector store instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        config = get_config()
        provider = config.VECTOR_DB_PROVIDER.lower()

        if provider == "qdrant":
            _vector_store_instance = QdrantAdapter()
        elif provider == "pinecone":
            _vector_store_instance = PineconeAdapter()
        elif provider == "memory":
            _vector_store_instance = InMemoryVectorAdapter()
        else:
            _vector_store_instance = ChromaDBAdapter()
        logger.info(f"Initialized VectorStoreAdapter: {provider}")

    return _vector_store_instance
