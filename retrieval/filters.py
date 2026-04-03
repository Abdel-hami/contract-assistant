from langchain_classic.chains.query_constructor.base import load_query_constructor_runnable, load_query_constructor_chain
from langchain_community.query_constructors.qdrant import QdrantTranslator
from langchain_classic.chains.query_constructor.ir import Comparison, Operation
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from qdrant_client import models
import os
from langchain_groq import ChatGroq
def flatten_metadata_values(node):
    """
    Recursively finds dictionaries like {'date': '...'} in the AST 
    and replaces them with the actual string value.
    """
    if isinstance(node, Comparison):
        # If the value is that problematic dictionary, grab just the date string
        if isinstance(node.value, dict) and 'date' in node.value:
            node.value = node.value['date']
            
    elif isinstance(node, Operation):
        # Recursively check all arguments in AND/OR operations
        for arg in node.arguments:
            flatten_metadata_values(arg)
    return node
def remove_content_prefix(node):
    """
    Recursively removes 'content.' from any attribute names in the AST.
    """
    if hasattr(node, 'attribute') and node.attribute.startswith("content."):
        node.attribute = node.attribute.replace("content.", "")
            
    if hasattr(node, 'arguments'):
        for arg in node.arguments:
            remove_content_prefix(arg)
    return node

def clean_qdrant_filters(filter_obj):
    """
    Recursively removes 'content.' and any leading dots from 
    FieldCondition keys to match a flat Qdrant payload.
    """
    if filter_obj is None:
        return None

    # Helper to clean a list of conditions (must, should, must_not)
    def clean_list(conditions):
        if not conditions:
            return
        for condition in conditions:
            # 1. If it's a direct FieldCondition, fix the key
            if isinstance(condition, models.FieldCondition):
                # Remove 'content.' if present
                new_key = condition.key.replace("content.", "")
                # Remove a leading dot if it exists (e.g., '.party_1' -> 'party_1')
                if new_key.startswith("."):
                    new_key = new_key[1:]
                condition.key = new_key
                
            # 2. If it's a nested Filter (Filter inside a Filter), recurse
            elif hasattr(condition, 'must') or hasattr(condition, 'should'):
                clean_qdrant_filters(condition)

    # Apply to all possible Qdrant filter branches
    clean_list(filter_obj.must)
    clean_list(filter_obj.should)
    clean_list(filter_obj.must_not)
                
    return filter_obj
def get_filter_from_query(query: str): 
    metadata_info = [
        AttributeInfo(
            name="source_file",
            description="The name of the document. Use this to filter by name of document",
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
    
    # Enable JSON mode for reliable structured output
    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        groq_api_key=groq_api_key,
        temperature=0,
        # model_kwargs={
        #     "response_format": {"type": "json_object"}  # Force JSON output
        # }
    )
    constractor_chain = load_query_constructor_runnable(
        llm,
        document_content_description,
        metadata_info
    )
    result = constractor_chain.invoke({"query": query})
    print(f"result: {result}")
    translator = QdrantTranslator(metadata_key="")
    
    if result.filter:
        try:
            clean_filter = flatten_metadata_values(result.filter)
            print(f"clearn_filter: {clean_filter}")
            qdrant_filter = translator.visit_operation(clean_filter)
            qdrant_filter = clean_qdrant_filters(qdrant_filter)
        except Exception as e:
            print(f"Translation failed: {e}")
            qdrant_filter = None
    else:
        qdrant_filter = None

    return qdrant_filter
