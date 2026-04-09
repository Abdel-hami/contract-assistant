from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os 

load_dotenv()

class QueryRewriting:
    def __init__(self, model_name:str="llama-3.3-70b-versatile"): #
        self.model_name = model_name

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.llm = ChatGroq(groq_api_key = groq_api_key, model_name = self.model_name, temperature=0)

        self.prompt = PromptTemplate(
            input_variables=["original_query"],
            template="""You are an expert legal query rewriting and expansion assistant.

            Your task is to rewrite the user's query into a retrieval-optimized search query for legal contracts.

            Rules:
            - Preserve the original legal intent exactly (DO NOT add new concepts)
            - Expand the query by at most five words without changing the legal meaning or terminology.            
            - Avoid document-specific names, metadata, or filters
            - Output a single query

            Original query: {original_query}

            Rewritten query:"""
        )
    
        self.chain = self.prompt | self.llm
        print(f"[INFO] Groq LLM initialized {self.model_name}")

    def rewrite_query(self,original_query:str):
        if not original_query.strip():
            raise ValueError("original query cannot be empty")
        response = self.chain.invoke({"original_query": original_query})
        return response.content.strip()
