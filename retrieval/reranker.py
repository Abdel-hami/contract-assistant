# In a RAG pipeline, relevant documents from a knowledge base are identified and 
# retrieved based on how similar they are to the user’s query. 
# More specifically, the similarity of each text chunk is quantified using a retrieval metric,
#
# Unfortunately, high similarity scores don’t always guarantee perfect relevance.
# In other words, the retriever may retrieve a text chunk that has a high similarity score,
# but is in fact not that useful – just not what we need to answer our user’s question 🤷🏻‍♀️.
# And this is where re-ranking is introduced, as a way to refine results before feeding them into the LLM.

##

# So, this is the issue we try to tackle by introducing the reranking step. In essence,
#  reranking means re-evaluating the chunks that 
# are retrieved based on the cosine similarity scores with a more accurate, yet also more expensive and slower method.


# Why reranking?
#     Hybrid search retrieves the top-20 CANDIDATES efficiently.
#     But vector similarity ≠ actual relevance to the query.
#     A reranker reads query + chunk TOGETHER and scores true relevance.
#     This is the single biggest quality improvement in a RAG pipeline.

import logging
from dataclasses import dataclass
import os 
from dotenv import load_dotenv
import cohere

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class RerankedResult:
    """
    A chunk after reranking — carries the rerank score alongside
    all original SearchResult fields.

    Attributes:
        chunk_id       : unique chunk identifier
        text           : contract chunk text
        rerank_score   : relevance score from reranker (higher = more relevant)
        section_title  : section in the contract this chunk came from
        metadata       : full payload (contract_type, expiry_date, file_name..)
        original_rank  : position before reranking (from hybrid search)
    """
    chunk_id: str
    rerank_score: float
    metadata: dict
    original_rank: int


class Reranker:
    # rerank-v3.5 / rerank-v4.0
    #rerank-english-v3.0
    def __init__(self, model_name:str = "rerank-v3.5"):
        self.model_name = model_name
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("cohere api key not found in the .env file")
        self.client = cohere.Client(cohere_api_key)
        logger.info(f"Reranker initialized - model: {model_name}")

    def rerank(self, query: str, results: list, top_n:int=10)->list:
        """
        Rerank hybrid search results using Cohere API. 
        Returns:
            list of RerankedResult sorted by rerank_score descending
        """
        logger.info(f"Start Reranking ...")
        if not results:
            return []
        documents = [r.metadata.get("text","") for r in results] # r.page_content for r in results

        response = self.client.rerank(
            model = self.model_name,
            query = query,
            documents = documents,
            top_n = top_n,
            return_documents = False
        )

        reranked = []
        for hit in response.results:
            original = results[hit.index]
            reranked.append(RerankedResult(
                chunk_id=original.chunk_id,
                rerank_score=hit.relevance_score,
                metadata=original.metadata,
                original_rank=hit.index + 1,
            ))
        logger.info(f"Cohere reranked {len(results)} → top {len(reranked)} results")

        return reranked


