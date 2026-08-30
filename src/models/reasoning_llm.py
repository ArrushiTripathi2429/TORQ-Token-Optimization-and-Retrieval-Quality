"""
src/models/reasoning_llm.py

Wrapper around the main reasoning LLM — the larger, more capable (and
more expensive) model that actually generates the final answer using
the assembled, budget-compliant context.

This is intentionally a thin wrapper: swap the provider (Gemini, GPT,
Claude, etc.) here without touching reasoning_node.py or any other
part of the graph. Keeping the actual API call isolated to one file
is what makes it easy to A/B test models later for your cost/quality
comparison in results/.
"""

import os
from groq import Groq

# Move to config.py once that exists.
# llama-3.3-70b-versatile is Groq's larger, more capable model —
# appropriate for the main reasoning step. See quality_eval_node.py
# for the cheaper model used for judging, kept deliberately separate.
REASONING_MODEL_NAME = "llama-3.3-70b-versatile"

_client = None


def _load_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to your .env file — "
                "never hardcode API keys directly in source files."
            )
        _client = Groq(api_key=api_key)
    return _client


PROMPT_TEMPLATE = """You are answering a question using ONLY the context provided below. \
If the context does not contain enough information to answer confidently, say so explicitly \
rather than guessing. Cite the source document name when you use a specific fact.

Context:
{context}

Question: {query}

Answer:"""


def generate_answer(query: str, context: str) -> str:
    """
    Sends the assembled context + query to the reasoning LLM and
    returns the generated answer as plain text.
    """
    client = _load_client()
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    response = client.chat.completions.create(
        model=REASONING_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # low temperature — factual grounding matters more than creativity here
    )
    return response.choices[0].message.content.strip()