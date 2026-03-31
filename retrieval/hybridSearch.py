## Hybrid search ---> a technique that combine between dense vectore Search (for semantic understanding) 
# and sparse vector search (for keyword matching) to deliever more accurate and relevant results.
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter
from fastembed import SparseTextEmbedding
from typing import Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


# result schema
@dataclass
class SearchResult:
    chunk_id:str
    score:float
    metadata:dict 


class HybridSearch:
    def __init__(self, dense_model_name:str= "BAAI/bge-small-en-v1.5", sparse_model_name:str="Qdrant/bm25"):
        self.dense_model_name = dense_model_name
        self.dense_model = SentenceTransformer(self.dense_model_name)
        self.sparse_model_name = sparse_model_name
        self.sparse_model = SparseTextEmbedding(self.sparse_model_name)

        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "contracts"
        
        logger.info(f"Hybrid Search engine initialized with dense model: {dense_model_name} and sparse model: {sparse_model_name}")

    def hybrid_search_with_rrf(self, query:str, filters:Optional[Filter]=None):
        """Perform hybrid search using Reciprocal Rank Fusion"""
        # rrf is a method for merging and ranking results from multiple search techniques

        logger.info("Start Hybrid search ...")
        query_dense_embedding = self.dense_model.encode(query,normalize_embeddings=True)
        query_sparse_embeding = list(self.sparse_model.embed(query))[0]

        query_sparse_vector = models.SparseVector(
            indices=query_sparse_embeding.indices,
            values=query_sparse_embeding.values
        )

        ## hybrid search
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense_embedding,
                    using="dense",
                    limit=10,
                    filter=filters
                ),
                models.Prefetch(
                    query=query_sparse_vector,
                    using="sparse",
                    limit=10,
                    filter=filters
                )
            ],
            query= models.FusionQuery(fusion=models.Fusion.RRF)
        )
        logger.info("Found results successfully.")
        return self.hybrid_search_points_to_results(results.points)
    
    def hybrid_search_points_to_results(self, points):
        return [
            SearchResult(
                chunk_id=str(point.id),
                score = point.score,
                metadata=point.payload
            )
            for point in points
        ]



# if __name__ == "__main__":
#     hybrid_search = HybridSearch()
#     results = hybrid_search.hybrid_search_with_rrf("an exanple query:")

#     for i, point in enumerate(results,1):
#         print(f"{i}. {point.payload.get('title', 'No title')} (Score: {point.score:.3f})")






