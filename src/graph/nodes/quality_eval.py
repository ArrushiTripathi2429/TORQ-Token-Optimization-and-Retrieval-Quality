"""
src/graph/nodes/quality_eval_node.py

Node 6 of the TORQ graph: Quality Evaluation.

Scores the generated answer for quality, and updates retry_count if
this is a retry attempt. The actual accept/retry DECISION is made by
edges.py's route_after_quality — this node only measures and reports
a score, same separation-of-concerns pattern as budget_check_node.

Uses an LLM-as-judge approach: a cheap/fast model reads the query,
context, and answer, and rates how well the answer is supported by
the context and how directly it addresses the query. This is more
meaningful than a simple heuristic (like answer length) because it
actually checks groundedness — whether the answer is hallucinating
beyond what the context supports.
"""

import os
import re
import google.generativeai as genai

from src.graph.state import RAGState

# Cheap/fast model for judging — deliberately NOT the same (expensive)
# model used for reasoning_node, to keep the quality-check cost low.
# Move to config.py once that exists.
JUDGE_MODEL_NAME = "gemini-1.5-flash"

_judge_model = None


def _load_judge():
    global _judge_model
    if _judge_model is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set. Add it to your .env file.")
        genai.configure(api_key=api_key)
        _judge_model = genai.GenerativeModel(JUDGE_MODEL_NAME)
    return _judge_model


JUDGE_PROMPT_TEMPLATE = """Rate the following answer on a scale of 0.0 to 1.0 based on:
1. Is the answer directly supported by the given context (no hallucination)?
2. Does it actually address the question asked?

Respond with ONLY a number between 0.0 and 1.0. No explanation, no other text.

Context:
{context}

Question: {query}

Answer: {answer}

Score:"""


def _parse_score(raw_response: str) -> float:
    """
    Extracts a float from the judge model's response. LLMs don't always
    follow "respond with only a number" perfectly, so this pulls the
    first number-like pattern out rather than assuming a clean response.
    Falls back to 0.0 (fail-safe: treat unparseable judgments as low
    quality, which triggers a retry rather than silently accepting a
    possibly-bad answer).
    """
    match = re.search(r"(\d+\.?\d*)", raw_response)
    if match:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))  # clamp to valid range
    return 0.0


def quality_eval_node(state: RAGState) -> dict:
    """
    Judges the generated answer against the context and query,
    returns a quality_score between 0.0 and 1.0.
    """
    judge = _load_judge()
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        context=state["final_context"],
        query=state["query"],
        answer=state["answer"],
    )

    response = judge.generate_content(prompt)
    score = _parse_score(response.text)

    print(f"Quality eval: score = {score:.2f}")

    return {"quality_score": score}