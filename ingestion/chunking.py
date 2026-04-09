import logging
import re
from typing import List
from langchain_core.documents import Document

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
        header_info = f"Section {match.group(0).strip()}:\n"

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
                # For massive sections, split with OVERLAP so context isn't lost at the cut
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200, 
                    add_start_index=True
                )
                splits = text_splitter.split_text(section)
                for split in splits:
                    all_chunks.append(Document(page_content=split, metadata=doc.metadata.copy()))



    all_chunks =  [doc for doc in all_chunks if len(doc.page_content) > 30]
    logger.info(f"Created {len(all_chunks)} chunks")
    return all_chunks
