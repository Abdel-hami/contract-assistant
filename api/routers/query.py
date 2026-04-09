from pydantic import BaseModel
from typing import Any
from fastapi import Request, APIRouter
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


    return QueryResponse(
        answer=result["answer"],
        sources = result["sources"],
    )
