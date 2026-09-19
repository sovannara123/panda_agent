import pytest
import os
import uuid
from panda_agent.rag.vector_store import (
    InMemoryVectorAdapter,
    ChromaDBAdapter,
    get_vector_store
)
from panda_agent.core.config import get_config


def test_in_memory_vector_adapter():
    adapter = InMemoryVectorAdapter()
    
    ids = ["doc1_chunk0", "doc1_chunk1"]
    chunks = ["Python is a programming language.", "FastAPI is a web framework for Python."]
    embeddings = [[0.1, 0.2, 0.3], [0.1, 0.25, 0.35]]
    metadatas = [{"source": "test.txt", "page_number": 1}, {"source": "test.txt", "page_number": 2}]
    
    adapter.add_chunks(ids=ids, chunks=chunks, embeddings=embeddings, metadatas=metadatas)
    
    # Query
    results = adapter.query(query_embedding=[0.1, 0.2, 0.3], top_k=2)
    assert len(results) == 2
    assert "Python" in results[0]["chunk"]
    
    # List docs
    docs = adapter.list_documents()
    assert len(docs) == 1
    assert docs[0]["source"] == "test.txt"
    assert docs[0]["chunk_count"] == 2
    
    # Delete doc
    adapter.delete_document("test.txt")
    assert len(adapter.list_documents()) == 0


def test_chroma_db_adapter(tmp_path):
    config = get_config()
    orig_path = config.CHROMA_DB_PATH
    config.CHROMA_DB_PATH = str(tmp_path / "chroma_test")
    
    try:
        adapter = ChromaDBAdapter(collection_name=f"test_{uuid.uuid4().hex[:6]}")
        
        ids = ["c1"]
        chunks = ["RAG architecture test chunk."]
        embeddings = [[0.5, 0.5, 0.5]]
        metadatas = [{"source": "rag_test.pdf", "page_number": 1}]
        
        adapter.add_chunks(ids=ids, chunks=chunks, embeddings=embeddings, metadatas=metadatas)
        
        results = adapter.query(query_embedding=[0.5, 0.5, 0.5], top_k=1)
        assert len(results) == 1
        assert results[0]["chunk"] == "RAG architecture test chunk."
        
        docs = adapter.list_documents()
        assert len(docs) == 1
        assert docs[0]["source"] == "rag_test.pdf"
    finally:
        config.CHROMA_DB_PATH = orig_path


def test_factory_get_vector_store():
    config = get_config()
    orig_provider = config.VECTOR_DB_PROVIDER
    try:
        config.VECTOR_DB_PROVIDER = "memory"
        import panda_agent.rag.vector_store as vs_module
        vs_module._vector_store_instance = None
        store = get_vector_store()
        assert isinstance(store, InMemoryVectorAdapter)
    finally:
        config.VECTOR_DB_PROVIDER = orig_provider
        import panda_agent.rag.vector_store as vs_module
        vs_module._vector_store_instance = None
