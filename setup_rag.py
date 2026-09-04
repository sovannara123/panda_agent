from rag import ingest_documents
from sample_docs import DOCUMENTS_TO_INGEST

if __name__ == "__main__":
    print("Ingesting documents into knowledge base...")
    ingest_documents(DOCUMENTS_TO_INGEST, source="panda_agent_docs")
    print("Ingestion complete! You can now ask the agent about these documents.")