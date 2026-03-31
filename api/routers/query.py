from pydantic import BaseModel
from typing import Any
from fastapi import Request, APIRouter, HTTPException
# from fastapi.responses import StreamingResponse
from generation.question_answering import build_user_prompt, SYSTEM_PROMPT
from retrieval.filters import get_filter_from_query
# from api.streaming import stream_answer
import logging


logger = logging.getLogger(__name__)


router = APIRouter(tags=["Query"])

## request / response schema


class QueryRequest(BaseModel):
    query: str
    top_k: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[Any]

# ------
# def run_pipeline(request: Request, query: str, top_k: int = 5):
#     """Runs the full RAG pipeline:
#     rewrite -> filter -> search -> rerank -> returns chunks + rewriting query"""

#     ##This is used to store data that should be accessible throughout the entire lifecycle of the application,
#     # such as database connection pools or machine learning models

#     state = request.app.state

#     # 1- rewriter
#     rewritten = state.rewriter.rewrite_query(query)

#     # 2-extract metadata filters from original query
#     qdrant_filters = get_filter_from_query(query)

#     # 3- hybrid search
#     results = state.search.hybrid_search_with_rrf(
#         rewritten, top_n=top_k * 4, filters=qdrant_filters
#     )
#     if not results:
#         raise HTTPException(status_code=404, detail="no relevant contracts found")

#     # 4 - rerank chunks
#     reranked_results = state.reranker.rerank(query, results, top_n=top_k)

#     # 5 - convert to chunk dict for llm
#     chunks = state.llm.reranked_to_chunks(reranked_results)

#     return {"rewritten_query": rewritten, "chunks": chunks}


@router.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest, request: Request): 
    """full RAG pipeline - returns a json answer"""

    logger.info(f"Query recieved succussfully - {body.query}")
    pipeline = request.app.state.ragPipeline
    run = pipeline.run(body.query,body.top_k)
    # generate response
    result = pipeline.llm.generate_response(
        query=run["rewritten_query"], chunks=run["chunks"]
    )

    ## to - do later
    print(result)
    # return QueryResponse(
    #     answer=result.answer,
    #     sources = result.sources,
    # )

@router.get("\home")
def home():
    return {"message":"hello"}

print("end")
# @router.get("/query/streaming")
# def streaming_asnwer_endpoint(
#     query: str,
#     request: Request,
#     top_k: int = 5,
# ):  ##key:Depens(verify_api_key)
#     """
#     Streaming RAG pipeline — returns tokens via SSE as they are generated.
#     """
#     logger.info(f"query recieved - {query}")
#     pipeline = request.app.state.ragPipeline
#     run = pipeline.run(query,top_k)
#     user_prommt = build_user_prompt(run["query"], run["chunks"])

#     return StreamingResponse(
#         stream_answer(
#             client=pipeline.llm.client,
#             model_name=pipeline.llm.model_name,
#             system_prompt=SYSTEM_PROMPT,
#             user_prompt=user_prommt,
#         ),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "X-Accel-Buffering": "no",  # disable nginx buffering
#         },
#     )


"""
api/routers/query.py
Contract Intelligence Platform — Query Router

Two endpoints:
    POST /api/v1/query          → full answer at once (JSON)
    GET  /api/v1/query/stream   → streaming answer token by token (SSE)
# """

# import logging
# from fastapi import APIRouter, Request, Depends, HTTPException
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel

# from api.auth.api_key import verify_api_key
# from api.streaming import stream_answer
# from retrieval.filters import get_filter_from_query
# from generation.guardrails import run_guardrails
# from generation.prompts.contract_qa import SYSTEM_PROMPT, build_user_prompt

# logger = logging.getLogger(__name__)
# router = APIRouter(tags=["Query"])


# # ── Request / Response schemas ────────────────────────────────────────────────


# class QueryRequest(BaseModel):
#     query: str  # user's question
#     top_k: int = 5  # number of chunks to retrieve


# class CitationResponse(BaseModel):
#     file_name: str
#     section_title: str | None = None
#     relevant_quote: str | None = None


# class QueryResponse(BaseModel):
#     answer: str
#     confidence: str
#     intent: str
#     citations: list[CitationResponse]
#     expiry_date: str | None = None
#     contract_value: str | None = None
#     hallucination_flag: bool = False
#     contains_pii: bool = False


# # ── Helper: run full retrieval pipeline ──────────────────────────────────────


# def run_pipeline(request: Request, query: str, top_k: int) -> dict:
#     """
#     Runs the full RAG pipeline:
#     rewrite → filter → search → rerank → return chunks + rewritten query
#     """
#     state = request.app.state

#     # Step 1 — rewrite query
#     rewritten = state.rewriter.rewrite(query)

#     # Step 2 — extract metadata filters from ORIGINAL query
#     qdrant_filter = get_filter_from_query(query)

#     # Step 3 — hybrid search
#     results = state.search.search(rewritten, top_n=top_k * 4, filters=qdrant_filter)
#     if not results:
#         raise HTTPException(status_code=404, detail="No relevant contracts found.")

#     # Step 4 — rerank
#     reranked = state.reranker.rerank(rewritten, results, top_n=top_k)

#     # Step 5 — convert to chunk dicts for LLM
#     from generation.llm_client import LLMClient

#     chunks = LLMClient.reranked_to_chunks(reranked)

#     return {"rewritten": rewritten, "chunks": chunks}


# # ── POST /query — full answer ─────────────────────────────────────────────────


# @router.post("/query", response_model=QueryResponse)
# async def query_endpoint(
#     body: QueryRequest,
#     request: Request,
#     _key: str = Depends(verify_api_key),
# ):
#     """
#     Full RAG pipeline — returns complete JSON answer.

#     Request:
#         POST /api/v1/query
#         Header: X-API-Key: your_key
#         Body: {"query": "when does the Armstrong IP agreement expire?"}

#     Response:
#         {
#             "answer": "The agreement expires on ...",
#             "confidence": "high",
#             "citations": [{"file_name": "Armstrong.pdf", ...}],
#             ...
#         }
#     """
#     logger.info(f"Query received: '{body.query}'")

#     pipeline = run_pipeline(request, body.query, body.top_k)

#     # Generate answer
#     answer = request.app.state.llm.generate(
#         query=pipeline["rewritten"],
#         chunks=pipeline["chunks"],
#     )

#     # Run guardrails
#     result = run_guardrails(answer, pipeline["chunks"])

#     return QueryResponse(
#         answer=result.answer.answer,
#         confidence=result.answer.confidence,
#         intent=result.answer.intent,
#         citations=[
#             CitationResponse(
#                 file_name=c.file_name,
#                 section_title=c.section_title,
#                 relevant_quote=c.relevant_quote,
#             )
#             for c in result.answer.citations
#         ],
#         expiry_date=result.answer.expiry_date,
#         contract_value=result.answer.contract_value,
#         hallucination_flag=result.hallucination_flag,
#         contains_pii=result.pii_flag,
#     )


# # ── GET /query/stream — streaming answer ──────────────────────────────────────


# @router.get("/query/stream")
# async def query_stream_endpoint(
#     query: str,
#     request: Request,
#     top_k: int = 5,
#     _key: str = Depends(verify_api_key),
# ):
#     """
#     Streaming RAG pipeline — returns tokens via SSE as they are generated.

#     Request:
#         GET /api/v1/query/stream?query=when+does+the+contract+expire
#         Header: X-API-Key: your_key

#     Response (SSE stream):
#         data: {"token": "The"}
#         data: {"token": " contract"}
#         data: {"token": " expires"}
#         ...
#         data: {"status": "done"}
#     """
#     logger.info(f"Stream query received: '{query}'")

#     pipeline = run_pipeline(request, query, top_k)

#     user_prompt = build_user_prompt(
#         query=pipeline["rewritten"],
#         chunks=pipeline["chunks"],
#     )

#     return StreamingResponse(
#         stream_answer(
#             client=request.app.state.llm.client,
#             model_name=request.app.state.llm.model_name,
#             system_prompt=SYSTEM_PROMPT,
#             user_prompt=user_prompt,
#         ),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "X-Accel-Buffering": "no",  # disable nginx buffering
#         },
#     )
