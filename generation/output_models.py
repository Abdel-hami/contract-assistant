from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
## enum

class ConfidenceLevel(str, Enum):
    HIGH   = "high"    
    MEDIUM = "medium" 
    LOW    = "low"

class QueryIntent(str, Enum):
    EXPIRY_DATE      = "expiry_date"       
    TERMINATION      = "termination"       
    LIABILITY        = "liability"         
    PAYMENT          = "payment"           
    GOVERNING_LAW    = "governing_law"     
    AUTO_RENEWAL     = "auto_renewal"      
    PARTIES          = "parties"           
    CONFIDENTIALITY  = "confidentiality"  
    GENERAL          = "general"           


## Cotation schema

class Citation:
    """
    Points the user to the exact source of the answer.
    Every answer MUST have at least one citation.
    """
    file_name: str = Field()
    relevant_quote: Optional[str] = Field(
        default=None,
        description="Short verbatim quote (max 100 chars) from the contract"
    )


## answer schema

 
class ContractAnswer(BaseModel):
    """
    Structured response returned by the LLM for every contract query.

    The LLM is instructed to populate all fields.
    Guardrails validates this before it reaches the API.
    """

    answer: str = Field()

    confidence: ConfidenceLevel = Field(
        description="How confident the LLM is based on the retrieved context"
    )

    citations: list[Citation] = Field(
        description="Source documents that support this answer (min 1 required)"
    )

    intent: QueryIntent = Field(
        default=QueryIntent.GENERAL,
        description="Detected intent of the user's query"
    )

    # Extracted structured data (when applicable)
    expiry_date: Optional[str] = Field(
        default=None,
        description="Extracted expiry date if query is about contract end date"
    )
    contract_value: Optional[str] = Field(
        default=None,
        description="Extracted contract value if mentioned in answer"
    )
    notice_period_days: Optional[int] = Field(
        default=None,
        description="Termination notice period in days if mentioned"
    )
    metadata:dict

    class Config:
        use_enum_values = True
