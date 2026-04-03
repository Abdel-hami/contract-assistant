from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from torch import chunk
from generation.question_answering import SYSTEM_PROMPT, build_user_prompt


load_dotenv()

class LLMClient:
# llama-3.3-70b-versatile
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        """Groq LLM client for contract question answering."""
        self.model_name = model_name
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("can't find the GROQ_API_KEY in .env")
        self.llm = ChatGroq(
            model=self.model_name, groq_api_key=groq_api_key, temperature=0
        )

        print(f"[INFO] LlmClient initialized - model {self.model_name}")

    def generate_response(self, query: str, chunks):
        USER_PROMPT = build_user_prompt(query, chunks)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT),
        ]
        try:
            answer = self.llm.invoke(messages)
        except Exception as e:
            raise RuntimeError(f"Groq API call failed: {e}") from e

        sources = [
            {
                "file_name": chunk.get("source_file", ""),
                "file_type": chunk.get("file_type", ""),
                "page": chunk.get("page", ""),
                "contract_type": chunk.get("contract_type", ""),
                "agreement_date": chunk.get("agreement_date", ""),
                "effective_date": chunk.get("effective_date", ""),
                "expiration_date": chunk.get("expiration_date", ""),
                "agreement_date_human_display": chunk.get(
                    "agreement_date_human_display", ""
                ),
                "effective_date_human_display": chunk.get(
                    "effective_date_human_display", ""
                ),
                "expiration_date_human_display": chunk.get(
                    "expiration_date_human_display", ""
                ),
                "party_1": chunk.get("party_1", ""),
                "party_2": chunk.get("party_2", ""),
                "notice_period_to_terminate": chunk.get(
                    "notice_period_to_terminate", ""
                ),
                "renewl_term": chunk.get("renewl_term", ""),
                "governing_law": chunk.get("governing_law", ""),
                "preview": chunk["text"][0:200] + "...",
            }
            for chunk in chunks
        ]

        results = {
            "answer": answer.content,
            "sources": sources,
        }
        return self.format_result(results)
        # return results

    @staticmethod
    def format_result(results):
    # results is likely a dict containing a list under "sources"
    # or just a list itself. Adjust accordingly:
        sources_input = results.get("sources", [])
        
        sources_map = {}

        for source in sources_input:
            filename = source.get("file_name", "unknown")
            
            if filename not in sources_map:
                # First time seeing this file → create the entry
                sources_map[filename] = {
                    "filename": filename,
                    "file_type": source.get("file_type", ""),
                    "page": source.get("page", ""),
                    "contract_type": source.get("contract_type", ""),
                    "agreement_date": source.get("agreement_date", ""),
                    "effective_date": source.get("effective_date", ""),
                    "expiration_date": source.get("expiration_date", ""),
                    "agreement_date_human_display": source.get("agreement_date_human_display", ""),
                    "effective_date_human_display": source.get("effective_date_human_display", ""),
                    "expiration_date_human_display": source.get("expiration_date_human_display", ""),
                    "party_1": source.get("party_1", ""),
                    "party_2": source.get("party_2", ""),
                    "pages": [source["page"]] if source.get("page") else [],
                    "notice_period_to_terminate": source.get("notice_period_to_terminate", ""),
                    "renewl_term": source.get("renewl_term", ""),
                    "governing_law": source.get("governing_law", ""),
                    "preview": source.get("preview", ""),
                }
            else:
                # Already seen this file → update the pages list
                existing = sources_map[filename]
                page = source.get("page")
                if page and page not in existing["pages"]:
                    existing["pages"].append(page)

        return {
            "answer": results["answer"],
            "sources": list(sources_map.values()),
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
            }
            for r in reranked_results
        ]
