from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from generation.question_answering import (SYSTEM_PROMPT, build_user_prompt)
import os 
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


load_dotenv()

class LLMClient:

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        """Groq LLM client for contract question answering."""
        self.model_name = model_name
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("can't find the GROQ_API_KEY in .env")
        self.llm = ChatGroq(
            model=self.model_name, groq_api_key=groq_api_key, temperature=0
        )

        logger.info(f"LLMClient initialized - model {self.model_name}")

    def generate_response(self, query: str, chunks):

        logger.info(f"start generating response for query : {query}")

        USER_PROMPT = build_user_prompt(query, chunks)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT),
        ]
        try:
            answer = self.llm.invoke(messages)
        except Exception as e:
            raise RuntimeError(f"Groq API call failed: {e}") from e

        sources = [{
            "filename": chunk.get("source", ""),
            "file_type": chunk.get("file_type", ""),
            "page": chunk.get("page", ""),
            "contract_type": chunk.get("contract_type", ""),
            "agreement_date": chunk.get("agreement_date", ""),
            "effective_date": chunk.get("effective_date", ""),
            "expiration_date": chunk.get("expiration_date", ""),
            "party_1": chunk.get("party_1","uknown"),
            "party_2": chunk.get("party_2","uknown"),
            "notice_period_to_terminate": chunk.get("notice_period_to_terminate","uknown"),
            "renewl_term": chunk.get("renewl_term","uknown"),
            "governing_law": chunk.get("governing_law","uknown"),
            "preview": chunk["text"][0:200]+ "..."
            # "clause_type": chunk.get("Clause_type", ""),
        } for chunk in chunks]

        results = {
            "answer": answer.content,
            "sources": sources,
            # "confidence":chunks[0].get("rerank_score",0) if chunks else None
        }
        logger.info("Response generated successfully.")
        return self.format_result(results)
    
    @staticmethod
    def format_result(results):
            sources_map = {}

            for source in results.get("sources",[]):
                filename = source.get("filename", "unknown")

                if filename not in sources_map:
                    # First time seeing this file → add it
                    sources_map[filename] = {
                        "filename": filename,
                        "contract_type": source.get("contract_type", ""),
                        "agreement_date": source.get("agreement_date", ""),
                        "effective_date": source.get("effective_date", ""),
                        "expiration_date": source.get("expiration_date", ""),
                        "party_1": source.get("party_1",""),
                        "party_2": source.get("party_2",""),
                        "notice_period_to_terminate": source.get("notice_period_to_terminate",""),
                        "renewl_term": source.get("renewl_term",""),
                        "pages": [source["page"]] if source.get("page") else [],
                        "preview": source.get("preview", ""),
                        # "clause_types":  [source.get("clause_type")] if source.get("clause_type") else [],
                    }
                else:
                    # Already seen this file → just add new page and clause_type
                    existing = sources_map[filename]

                    if source.get("page") and source["page"] not in existing["pages"]:
                        existing["pages"].append(source["page"])

                    # if source.get("clause_type") and source["clause_type"] not in existing["clause_types"]:
                    #     existing["clause_types"].append(source["clause_type"])

            return {
                "answer":     results["answer"],
                "sources":    list(sources_map.values()),
                # "confidence": results["confidence"],
            }

    # ── Chunk formatter ───────────────────────────────────────────────────────
    @staticmethod
    def reranked_to_chunks(reranked_results: list) -> list[dict]:
        """
        Convert RerankedResult objects from reranker.py into
        the simple dict format expected by build_user_prompt().

        Args:
            reranked_results : list of RerankedResult from Reranker.rerank()

        Returns:
            list of dicts with keys: text, file_name, section_title, chunk_id
        """
        return [
            {
                **r.metadata,
                "chunk_id": r.chunk_id,
                "text": r.metadata.get("text", ""),
                "file_name": r.metadata.get("source", "unknown"),
                "original_score": r.original_rank,
                "rerank_score": r.rerank_score
            }
            for r in reranked_results
        ]



# if __name__ == "__main__":
    
#     llm_client = LLMClient()
#     reranker = Reranker()
#     query_rewriter = QueryRewriting()
#     hybrid_search = HybridSearch()
#     original_query = "what is bla bla bla ?"

#     qdrant_filters = get_filter_from_query(original_query)
#     rewrited_query = query_rewriter.rewrite_query(original_query)
#     results = hybrid_search.hybrid_search_with_rrf(rewrited_query,fliters=qdrant_filters)
#     reranked_results = reranker.rerank(rewrited_query, results)
#     chunks = llm_client.reranked_to_chunks(reranked_results)

#     answer = llm_client.generate_response(rewrited_query,chunks)

#     print(answer)