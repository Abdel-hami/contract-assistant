#SSE (Server-Sent Events) streaming for Retrieval-Augmented Generation (RAG) is a technique used to deliver
# AI-generated responses to users in real-time, token-by-token, rather than waiting for the entire answer to be generated.

"""Contract Intelligence Platform — SSE Streaming Handler

SSE = Server-Sent Events
Instead of waiting for the full answer, the client receives tokens
one by one as the LLM generates them — feels much more responsive."""

import logging
import json
from groq import Groq

logger = logging.getLogger(__name__)

async def stream_answer(
        client: Groq,
        model_name:str,
        system_prompt:str,
        user_prompt:str
) :
    full_response = ""

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages={
                "role" : "system", "content" : system_prompt,
                "role" : "user", "content" : user_prompt
            },
            temperature=0.0,
            max_tokens=1024,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                #yield each token as sse event
                yield f"data: {json.dumps({"token":delta})}\n\n"
        # stream finished
        yield f"data: {json.dumps({"status":"done"})}"
        logger.info(f"Streamed response: {len(full_response)} chars")
    except Exception as e:
        logger.error(f"streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"