## creating data
import logging
import hashlib
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
import uuid

logger = logging.getLogger(__name__)


## embeed then store
class QdrantStore:

    def __init__(
        self,
        dense_model_name: str = "all-MiniLM-L6-v2",
        sparse_model_name: str = "Qdrant/bm25",
    ):
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.dense_model = SentenceTransformer(self.dense_model_name)
        self.saprse_model = SparseTextEmbedding(self.sparse_model_name)
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "contracts"
        self.creat_collection()

    def creat_collection(self):
        if not self.client.collection_exists(collection_name="contracts"):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
    ## generate IDs
    def generate_doc_id(self, source: str, content: str):
        unique_string = f"{source}:{content}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:32]

    def embedde_chunks_and_store(self, chunks):

        logger.info(f"generating embedding {len(chunks)} chunks ...")

        texts = [chunk.page_content for chunk in chunks]

        dense_embeddings = self.dense_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )
        sparse_embeddings = list(self.saprse_model.embed(texts))

        # Focuses on Semantic Direction: By removing the effect of vector magnitude, normalization ensures that similarity is based on semantic meaning rather than frequency or length.
        logger.info(f"embeddings shape: {dense_embeddings.shape}")

        points = []
        batch_size = 64

        for chunk, dense_embedding, sparse_embedding in zip(
            chunks, dense_embeddings, sparse_embeddings
        ):
            ## generating ID
            source = chunk.metadata.get("source_file","uknown")
            doc_id = self.generate_doc_id(source,chunk.page_content)

            ## getting general context
            contract_type =chunk.metadata.get("contract_type","Legal Document")
            parties = f"{chunk.metadata.get('party_1','')} and {chunk.metadata.get('party_2','')}"
            points.append(
                models.PointStruct(
                    # id=uuid.uuid4(),
                    id=doc_id,
                    vector={
                        "dense": dense_embedding.tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_embedding.indices,
                            values=sparse_embedding.values,
                        ),
                    },
                    payload={
                        "text": f"Document: {contract_type} between {parties}, \n Context: {chunk.page_content}",
                        ## filename
                        "source": chunk.metadata.get("source_file", ""),
                        "file_type": chunk.metadata.get("file_type", ""),
                        "page": chunk.metadata.get("page", ""),
                        "contract_type": chunk.metadata.get("contract_type", ""),
                        # "clause_type": chunk.metadata.get("Clause_type", ""),
                        "agreement_date":chunk.metadata.get("agreement_date", ""),
                        "effective_date": chunk.metadata.get("effective_date", ""),
                        "expiration_date": chunk.metadata.get("expiration_date", ""),
                        "party_1": chunk.metadata.get("party_1", ""),
                        "party_2": chunk.metadata.get("party_2", ""),
                        "notice_period_to_terminate": chunk.metadata.get("notice_period_to_terminate", ""),
                        "renewl_term": chunk.metadata.get("renewl_term", ""),
                        "governing_law": chunk.metadata.get("governing_law", ""),
                    },
                )
            )
            if len(points) > batch_size:
                self.client.upsert(collection_name=self.collection_name, points=points)
                logger.info(f"upsertted {len(points)}..")
                points = []
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"upsertted final {len(points)} points.")

        logger.info("chunks stored in Qdrant.")

