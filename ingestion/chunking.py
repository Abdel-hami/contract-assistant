import logging
import re
from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

SECTION_PATTERN = re.compile(r"\n\s*(\d+)\.\s+[A-Z][^\n]+")


def split_by_sections(text: str):
    """
    Split contract text by legal section headers like:
    1. Services
    2. Compensation
    """

    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return [text]
    sections = []
    for i, match in enumerate(matches):
        header_info = f"Section {match.group(0)}:\n"

        start = match.start()
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)
        section_content = text[start:end].strip()
        sections.append(f"{header_info}{section_content}")
    return sections


def chunk_contract_documents(documents: List[Document], embedding_model:str = "BAAI/bge-large-en-v1.5") -> List[Document]:
    """
    Chunk contracts using section-aware chunking,
    then semantic chunking for large sections.
    """
    ## add model_kwargs and encode_kwargs to force using GPU
    model_kwargs = {"device":"cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name = embedding_model ,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )

    semantic_chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    
    logger.info(f"initialise embedding model for Semantic Chunker.")
    all_chunks = []
    logger.info("start chunking ...")
    i = 0
    for doc in documents:
        i+=1
        logger.info(f"Chunking document {i}/{len(documents)}")
        text = doc.page_content
        # split by contract sections
        sections = split_by_sections(text)
        for section in sections:
            if len(section) < 2000:
                # small section → keep as one chunk
                chunk = Document(
                    page_content= section,
                    metadata=doc.metadata.copy()
                )
                all_chunks.append(chunk)
            else:
                # large section → semantic chunking
                semantic_chunks = semantic_chunker.create_documents(
                    [section],
                    [doc.metadata]
                )
                all_chunks.extend(semantic_chunks)


    all_chunks =  [doc for doc in all_chunks if len(doc.page_content) > 25]
    logger.info(f"Created {len(all_chunks)} chunks")
    return all_chunks
