from  contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routers import query
from fastapi.middleware.cors import CORSMiddleware
from pipeline import RAGPipeline

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app:FastAPI):
    """ Initialize heavy objects once at startup and store them in app.state
    so every request can reuse them without re-loading models."""

    logger.info("stating up Contract Intelligence Platform")


    app.state.ragPipeline = RAGPipeline()
    yield
    #In FastAPI, the lifespan context manager must use the yield keyword to separate the "startup" logic from the "shutdown" logic.
    logger.info("pipeline loaded successfully, API READY")

## main app

def create_app() -> FastAPI:

    app = FastAPI(
        title="Contract Intelligence Platform",
        description="RAG-powered contract Q&A API",
        version="1.0.0",
        lifespan=lifespan
    )
# CORS — restrict in production to your frontend domain
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )    
    app.include_router(query.router)


    return app

app = create_app()
