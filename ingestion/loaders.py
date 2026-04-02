from typing import List, Any
from langchain_core.documents import Document
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions
import logging
logger = logging.getLogger(__name__)


def load_all_document(data_dir:str)->List[Document]:
    """load all documents from data directory"""

    data_path = Path(data_dir).resolve()
    logger.info(f"data path: {data_path}")

    # configure docling pipeline
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CUDA
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )
    pdf_files = list(data_path.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} pdf files")
    documents = []

    for pdf_file in pdf_files:
        try:
            logger.info(f"Loading pdf: {pdf_file}")
            # convert with docling
            result = converter.convert(str(pdf_file))
            doc = result.document

            ## split by pages
            pages = sorted(doc.pages.keys()) if doc.pages else []

            if pages:
                for page in pages:
                    page_text = doc.export_to_markdown(page_no=page)
                    if not page_text.strip():
                        continue

                    documents.append(Document(
                        page_content=page_text,
                        metadata={
                            "file_type":     "pdf",
                            "source_file":   pdf_file.name,
                            "contract_type": pdf_file.parent.name,  # ← folder = contract type
                            "page":          page,
                            "source":        str(pdf_file),
                        }
                    ))
            else:
                # Fallback — export full document as one chunk if no pages
                full_text = doc.export_to_markdown()
                if full_text.strip():
                    documents.append(Document(
                        page_content=full_text,
                        metadata={
                            "file_type":     "pdf",
                            "source_file":   pdf_file.name,
                            "contract_type": pdf_file.parent.name,
                            "page":          0,
                            "source":        str(pdf_file),
                        }
                    ))

            logger.debug(f"Loaded {pdf_file.name}")
        except Exception as e:
            logger.error(f"Failed to load {pdf_file}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents")  
    return documents