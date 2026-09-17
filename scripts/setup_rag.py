from panda_agent.rag.pipeline import ingest_text
from scripts.sample_docs import DOCUMENTS_TO_INGEST

def setup_rag():
    print("Ingesting documents into knowledge base...")
    for i, doc_text in enumerate(DOCUMENTS_TO_INGEST):
        source = f"panda_agent_docs_{i}"
        ingest_text(doc_text, source_name=source)
        print(f"  Ingested: {source}")
    print("Ingestion complete! You can now ask the agent about these documents.")


if __name__ == "__main__":
    setup_rag()