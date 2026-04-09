from qdrant_client import models
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, Filter
from fastembed import SparseTextEmbedding
from typing import Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

# result schema
@dataclass
class SearchResult:
    chunk_id:str
    context_header:str
    metadata:dict 


class HybridSearch:
    def __init__(self, dense_model_name:str= "BAAI/bge-large-en-v1.5", sparse_model_name:str="Qdrant/bm25"):
        self.dense_model_name = dense_model_name
        self.dense_model = SentenceTransformer(self.dense_model_name)
        self.sparse_model_name = sparse_model_name
        self.sparse_model = SparseTextEmbedding(self.sparse_model_name)

        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        self.client = QdrantClient(host=host, port=port)

        # self.client = QdrantClient(host="localhost", port=6333)

        self.collection_name = "contracts"
        
    def hybrid_search_with_rrf(self, query:str, filters:Optional[Filter]=None):
        """Perform hybrid search using Reciprocal Rank Fusion"""
        # rrf is a method for merging and ranking results from multiple search techniques
        query_dense_embedding = self.dense_model.encode(query,normalize_embeddings=True,batch_size=16)
        query_sparse_embeding = list(self.sparse_model.embed(query))[0]

        query_sparse_vector = models.SparseVector(
            indices=query_sparse_embeding.indices,
            values=query_sparse_embeding.values
        )
        ## hybrifd search
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=query_dense_embedding,
                    using="dense",
                    limit=10,
                    filter=filters
                ),
                Prefetch(
                    query=query_sparse_vector,
                    using="sparse",
                    limit=10,
                    filter=filters
                )
            ],
            query= FusionQuery(fusion=Fusion.RRF)
        )
        return self.hybrid_search_points_to_results(results.points)
    
    def hybrid_search_points_to_results(self, points):

        return [
            SearchResult(
                chunk_id=str(point.id),
                context_header = str(point.payload.get("context_header","")),
                metadata=point.payload
            )
            for point in points
        ]
