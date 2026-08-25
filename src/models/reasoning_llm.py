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
import google.generativeai as genai

# Move to config.py once that exists.
REASONING_MODEL_NAME = "gemini-1.5-pro"

_model = None


def _load_model():
    global _model
    if _model is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Add it to your .env file — "
                "never hardcode API keys directly in source files."
            )
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(REASONING_MODEL_NAME)
    return _model


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
    model = _load_model()
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    response = model.generate_content(prompt)
    return response.text.strip()