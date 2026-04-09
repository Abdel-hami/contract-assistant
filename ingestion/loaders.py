import logging
logger = logging.getLogger(__name__)

from typing import List, Any
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
import unicodedata

def clean_text(text):
    # replace \u00a0 with a standard space and handles other oddities
    text = unicodedata.normalize("NFKD", text)
    # Replace multiple spaces with a single space
    return " ".join(text.split())

# Apply this to your contract text before chunking
def load_all_document(data_dir:str)->List[Any]:
    """Load all documents from data directory"""
    data_path = Path(data_dir).resolve()
    logger.debug(f"data path: {data_path}")
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
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["file_type"] = "pdf"
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata["contract_type"] = pdf_file.parent.name
            logger.debug(f"loader {len(docs)} docs from pdf: {pdf_file}")
            documents.extend(docs)
            logger.debug(f"loaded document: {pdf_file}")
        except Exception as e:
            logger.error(f"Failed to load {pdf_file}: {e}")
    logger.info(f"loaded {len(documents)}")
    return documents

