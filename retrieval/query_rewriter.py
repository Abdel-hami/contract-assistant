# A PromptTemplate is a predefined, reusable structure for generating prompts for 
# Large Language Models (LLMs). Instead of manually writing a full prompt every time, 
# a template uses placeholders (variables) that are dynamically filled with specific data at runtime, 
# ensuring consistency, efficiency, and better control over AI outputs.

import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os 


load_dotenv()
logger = logging.getLogger(__name__)
class QueryRewriting:
    def __init__(self, model_name:str="allam-2-7b"): 
        self.model_name = model_name

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.llm = ChatGroq(groq_api_key = groq_api_key, model_name = self.model_name, temperature=0)

        self.prompt = PromptTemplate(
            input_variables=["original_query"],
            template="""You are an expert legal assistant helping retrieve information \
                    from enterprise contracts.

                    Reformulate the user query below to be more precise and retrieval-friendly.
                    Focus on: legal terminology, clause names, date references.
                    Return ONLY the rewritten query, nothing else.

                    Original query: {original_query}
                    Rewritten query:"""
        )

        self.chain = self.prompt | self.llm
        logger.info(f"Groq LLM initialized {self.model_name}")

    def rewrite_query(self,original_query:str):
        if not original_query.strip():
            raise ValueError("original query cannot be empty")
        response = self.chain.invoke({"original_query": original_query})
        logger.info(f"Query rewrited successfully - {response}")

        return response.content.strip()


# if __name__ == "__main__":
#     query_rewriting = QueryRewriting()

#     print(query_rewriting.rewrite_query("when the contract end ?"))