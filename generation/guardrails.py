## what are guardrails: -> serve to moderate the user interactions with the model
## safety mechanisms that intercept input and output at critical points in the RAG workflow
## Guardrails are essential for ensuring the safety and reliability of AI systems.By setting clear boundaries, we can prevent the model from generating harmful or misleading content. 

"""
guardials - Validate the llm output before it reaches the User

Two checks:
    1. Hallucination detection  → answer claims something not in context
    2. PII detection            → answer leaks personal data
    PII ( Personally Identifiable Information)

"""
