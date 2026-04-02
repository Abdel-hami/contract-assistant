from ingestion.loaders import load_all_document
from ingestion.metadata import enrich_metadata
from ingestion.chunking import chunk_contract_documents
from ingestion.vectorStore import QdrantStore


class IngestionPipeline:
    def __init__(self, dense_model_name: str = "all-MiniLM-L6-v2", sparse_model_name: str = "Qdrant/bm25"):
        self.qdrantStore = QdrantStore(dense_model_name=dense_model_name, sparse_model_name=sparse_model_name)

    def run(self, data_dir):
        documents = load_all_document(data_dir)
        documents = enrich_metadata(documents=documents, data_csv_path="data/full_contract_pdf")
        chunks = chunk_contract_documents(documents=documents)
        self.qdrantStore.embedde_chunks_and_store(chunks)

