## creating data
import logging
import hashlib
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


## embeed then store
class QdrantStore:

    def __init__(
        self,
        dense_model_name: str = "BAAI/bge-large-en-v1.5",
        sparse_model_name: str = "Qdrant/bm25",
    ):
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        ## add model_kwargs={'device': 'cuda'} for GPU Enabling
        self.dense_model = SentenceTransformer(self.dense_model_name,device="cuda")
        ## add providers=["CUDAExecutionProvider"] for GPU Enabling
        self.sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name, providers=["CUDAExecutionProvider"])
        self.client = QdrantClient(host="localhost", port=6333, timeout=60)
        self.collection_name = "contracts"
        self.creat_collection()

    def creat_collection(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=1024, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
                timeout=60
            )
    ## generate IDs
    def generate_doc_id(self, source: str, content: str):
        unique_string = f"{source}:{content}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:32]

    def embedde_chunks_and_store(self, chunks):

        logger.info(f"generating embedding {len(chunks)} chunks ...")

        texts = [chunk.page_content for chunk in chunks]

        dense_embeddings = self.dense_model.encode(
            texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True
        )
        sparse_embeddings = list(self.sparse_model.embed(texts))

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
            #context header
            contract_info = f"Type: {chunk.metadata.get('contract_type', 'Legal')}"
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
                        "text": chunk.page_content,
                        "context_header": f"{source} from {contract_info}",
                        **chunk.metadata,
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
