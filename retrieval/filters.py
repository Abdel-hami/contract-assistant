"""
retrieval/filters.py
Contract Intelligence Platform — Metadata Pre-Filter Builder

Contract type comes directly from the PDF's parent folder name:
    full_contract_pdf/Part_I/IP/Armstrong.pdf       → contract_type = "IP"
    full_contract_pdf/Part_I/Service/Birch.pdf      → contract_type = "Service"
    full_contract_pdf/Part_I/License_Agreements/... → contract_type = "License_Agreements"

No regex guessing needed — the folder IS the label.
"""

from ingestion.vectorStore import QdrantStore
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_groq import ChatGroq
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def get_filter_from_query(query:str):
    metadata_info = [
        AttributeInfo(
            name="source_file",
            description="The name of the document such as 'CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement.pdf'. Use this to filter by name of document",
            type="string"
        ),
        AttributeInfo(
            name="party_1",
            description="The name of the first legal entity or company mentioned in the contract. Use this to filter by the primary party or signer.",
            type="string"
        ),
        AttributeInfo(
            name="party_2",
            description="The name of the second legal entity or counterparty in the contract. Use this to filter for the second participant in the agreement.",
            type="string"
        ),
        AttributeInfo(
            name="contract_type",
            description="The category of the agreement, such as 'Franchise Agreement', 'NDA', 'Service Agreement', or 'Lease'. Use this when the user specifies a document type.",
            type="string"
        ),
        AttributeInfo(
            name="agreement_date",
            description="The date the contract was signed or formally created. Use for questions about when a contract was made.",
            type="string"
        ),
        AttributeInfo(
            name="effective_date",
            description="The official start date of the contract's obligations. Use for questions about when an agreement becomes active.",
            type="string"
        ),
        AttributeInfo(
            name="expiration_date",
            description="The date the contract ends or becomes invalid. Use for questions about renewals, terminations, or end dates.",
            type="string"
        ),
        AttributeInfo(
            name="governing_law",
            description="The legal jurisdiction or state laws that apply to the contract (e.g., 'California', 'New York', 'Morocco'). Use this when the user mentions a specific location or law.",
            type="string"
        ),
    ]


    document_content_description = "Detailed clauses and legal text from corporate contracts"

    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model_name = "", groq_api_key = groq_api_key,temperature=0)
    vectorestore = QdrantStore()

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore = vectorestore,## to-do later
        document_contents=document_content_description,
        metadata_field_info=metadata_info
    )

    result = retriever.invoke(query)

    filters = models.Filter(
        must=[
            models.FieldCondition(key="source_file", match=models.matchValue(result.source_file)),
            models.Filter(
                should=[
                    models.FieldCondition(key="party_1", match=models.MatchValue(result.party_1)),
                    models.FieldCondition(key="party_2", match=models.MatchValue(result.party_2))
                ]
            ),
            models.FieldCondition(key="contract_type", match=models.matchValue(result.contract_type)),
            models.FieldCondition(key="agreement_date", match=models.matchValue(result.agreement_date)),
            models.FieldCondition(key="effective_date", match=models.matchValue(result.effective_date)),
            models.FieldCondition(key="expiration_date", match=models.matchValue(result.expiration_date)),
            models.FieldCondition(key="governing_law", match=models.matchValue(result.governing_law)),
        ]
    )

    return filters




#----------------------------------



import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)
from qdrant_client import models


# ── Known contract types from CUAD folder structure ──────────────────────────
# Key   = exact folder name in your dataset
# Value = query keywords that map to it

CUAD_CONTRACT_TYPES = {
    "IP":                      ["ip", "intellectual property", "patent", "trademark", "copyright"],
    "License_Agreements":      ["license", "licensing", "licence"],
    "Service":                 ["service", "services", "service agreement"],
    "NDA":                     ["nda", "non-disclosure", "confidentiality"],
    "Maintenance":             ["maintenance", "support", "upkeep"],
    "Development":             ["development", "dev", "software development"],
    "Distributor":             ["distributor", "distribution"],
    "Endorsement":             ["endorsement", "endorse"],
    "Franchise":               ["franchise", "franchising"],
    "Hosting":                 ["hosting", "host", "cloud hosting"],
    "Joint_Venture":           ["joint venture", "jv", "partnership"],
    "Manufacturing":           ["manufacturing", "manufacture"],
    "Marketing":               ["marketing", "advertisement", "advertising"],
    "Non_Compete_Non_Solicit": ["non-compete", "non compete", "non-solicit", "noncompete"],
    "Outsourcing":             ["outsourcing", "outsource", "bpo"],
    "Promotion":               ["promotion", "promotional"],
    "Reseller":                ["reseller", "resell", "resale"],
    "Sponsorship":             ["sponsorship", "sponsor"],
    "Strategic_Alliance":      ["strategic alliance", "alliance"],
    "Supply":                  ["supply", "supplier", "procurement"],
    "Transportation":          ["transportation", "transport", "logistics", "shipping"],
}


# ── 1. Used during INGESTION ──────────────────────────────────────────────────

def get_contract_type_from_path(file_path: str) -> str:
    """
    Extract contract type from the PDF's parent folder name.
   """
    contract_type = Path(file_path).parent.name
    logger.info(f"Contract type from path: '{contract_type}'")
    return contract_type


# ── 2. Used during RETRIEVAL ──────────────────────────────────────────────────

def detect_contract_type_from_query(query: str) -> Optional[str]:
    """
    Detect contract type from user query keywords.
    Maps to exact CUAD folder names.

    Args:
        query : raw or rewritten user query

    Returns:
        exact folder name or None if not detected

    Examples:
        "when does the IP agreement expire?"     → "IP"
        "what are the license terms?"            → "License_Agreements"
        "what is the liability cap?"             → None
    """
    query_lower = query.lower()
    for contract_type, keywords in CUAD_CONTRACT_TYPES.items():
        for keyword in keywords:
            if keyword in query_lower:
                logger.info(f"  Detected contract type: '{contract_type}'")
                return contract_type
    logger.info("  No contract type detected — searching all types")
    return None


def detect_party_from_query(query: str) -> Optional[str]:
    """
    Detect a company name from the query.

    Examples:
        "when does Armstrong's contract expire?" → "armstrong"
        "what is the liability cap?"             → None
    """
    stopwords = {
        "what", "when", "where", "which", "who", "how", "does", "do",
        "the", "this", "that", "is", "are", "was", "will", "can",
        "contract", "agreement", "clause", "section", "term", "terms",
        "expire", "expires", "expiry", "terminate", "termination",
    }
    capitalized = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', query)
    for word in capitalized:
        if word.lower() not in stopwords:
            logger.info(f"Detected party: '{word}'")
            return word.lower()
    return None


# ── Filter schema ─────────────────────────────────────────────────────────────

@dataclass
class ContractFilter:
    contract_type: Optional[str] = None   
    party_1:    Optional[str] = None   
    party_2:    Optional[str] = None   
    file_name:     Optional[str] = None   

    def is_empty(self) -> bool:
        return all(v is None for v in [self.contract_type, self.party_1, self.party_2, self.file_name])

# ── Extract + Build ───────────────────────────────────────────────────────────

def extract_filters(query: str) -> ContractFilter:
    """
    Extract all filters from a user query.

    Examples:
        "when does the Armstrong IP agreement expire?"
        → ContractFilter(contract_type="IP", party_name="armstrong")

        "what are the license renewal terms?"
        → ContractFilter(contract_type="License_Agreements")

        "what is the liability cap?"
        → ContractFilter()  ← empty, search everything
    """
    logger.info(f"Extracting filters from: '{query}'")
    filters = ContractFilter(
        contract_type=detect_contract_type_from_query(query),
        party_1=detect_party_from_query(query),
    )
    return filters


def build_qdrant_filter(filters: ContractFilter) -> Optional[Filter]:
    """ContractFilter → Qdrant Filter object (or None if empty)."""
    if filters.is_empty():
        return None

    conditions = []

    if filters.contract_type:
        conditions.append(FieldCondition(
            key="contract_type",
            match=MatchValue(value=filters.contract_type),
        ))
    if filters.party_1:
        conditions.append(FieldCondition(
            key="party_1",
            match=MatchValue(value=filters.party_1),
        ))
    if filters.party_2:
        conditions.append(FieldCondition(
            key="party_2",
            match=MatchValue(value=filters.party_2),
        ))
    if filters.file_name:
        conditions.append(FieldCondition(
            key="file_name",
            match=MatchValue(value=filters.file_name),
        ))

    return Filter(must=conditions)


def get_filter_from_query(query: str) -> Optional[Filter]:
    """
    One-shot: query string → Qdrant Filter.

    Usage:
        qdrant_filter = get_filter_from_query("Armstrong IP agreement expiry")
        results = search.search(query, filters=qdrant_filter)
    """
    return build_qdrant_filter(extract_filters(query))

