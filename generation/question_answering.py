from typing import List, Dict


SYSTEM_PROMPT = """You are a precise legal contract analyst assistant for an enterprise consulting firm.
Your job is to answer questions about contracts strictly based on the provided contract excerpts.

RULES YOU MUST FOLLOW:
1. ONLY answer using information found in the provided contract excerpts below.
2. If the answer is not in the excerpts, say exactly: "This information was not found in the provided contract documents."
3. NEVER invent dates, values, clause numbers, or party names.
5. Be concise and direct — answer in plain English, not legal jargon.
"""


def build_user_prompt(query:str, chunks:List[Dict])-> str:

    texts = [chunk.get("text","") for chunk in chunks]
    context = "\n\n".join(texts)
    return f"""{context}
    ---
    QUESTION: {query} 
    Answer based strictly on the excerpts above."""
