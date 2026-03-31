import logging
logger = logging.getLogger(__name__)

from typing import List, Any
from langchain_core.documents import Document
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

def load_all_document(data_dir:str)->List[Any]:
    """Load all documents from data directory"""
    data_path = Path(data_dir).resolve()
    logger.debug(f"data path: {data_path}")
    # print(f"[DEBUG] data path: {data_path}")
    documents = []

    ## PDFs
    pdf_files = list(data_path.glob("**/*.pdf"))
    logger.debug(f"Found {len(pdf_files)} pdf files")
    for pdf_file in pdf_files:
        try:
            logger.debug(f"Loading pdf: {pdf_file}")
            loader = PyMuPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["file_type"] = "pdf"
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata["contract_type"] = pdf_file.parent.name
                # print(doc.metadata)
            logger.debug(f"loader {len(docs)} docs from pdf: {pdf_file}")
            documents.extend(docs)
            logger.debug(f"loaded document: {pdf_file}")
        except Exception as e:
            logger.error(f"Failed to load {pdf_file}: {e}")
    logger.info(f"loaded {len(documents)}")
    return documents

