# pipeline.py

from retrieval.query_rewriter import QueryRewriting
from retrieval.hybridSearch  import HybridSearch
from retrieval.filters        import get_filter_from_query
from retrieval.reranker       import Reranker
from generation.llm_client    import LLMClient
from fastapi                  import HTTPException


class RAGPipeline:
    """
    Encapsulates the full RAG pipeline.
    Initialized once at startup, reused for every request.
    """

    def __init__(self):
        self.rewriter = QueryRewriting()
        self.search   = HybridSearch()
        self.reranker = Reranker() 
        self.llm      = LLMClient()

    def run(self, query: str, top_k: int = 8) -> dict:
        """
        Full RAG pipeline — no FastAPI dependency.
        Can be called from API, CLI, tests, anywhere.
        """

        # 1 — rewrite
        rewritten = self.rewriter.rewrite_query(query)

        # 2 — filters from original query
        qdrant_filters = get_filter_from_query(query)

        # if manual_filter:
        #     results = self.search.hybrid_search_with_rrf(rewritten,manual_filter)
        # else:
        #     results = self.search.hybrid_search_with_rrf(rewritten,qdrant_filters)

        results = self.search.hybrid_search_with_rrf(rewritten,qdrant_filters)
        # 3 — hybrid search
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No relevant contracts found."
            )

        # 4 — rerank
        reranked = self.reranker.rerank(query, results, top_n=top_k)

        # 5 — convert to chunks
        chunks = self.llm.reranked_to_chunks(reranked)

        return {"rewritten_query": rewritten, "chunks": chunks}
    

