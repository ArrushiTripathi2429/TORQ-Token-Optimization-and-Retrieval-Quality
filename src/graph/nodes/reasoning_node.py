"""
src/graph/nodes/reasoning_node.py

Node 5 of the TORQ graph: Main Reasoning.

Thin node — the actual LLM call logic lives in reasoning_llm.py.
This node's only job is to pull query + final_context out of state,
pass them to the LLM wrapper, and put the answer back into state.

Keeping nodes thin like this (delegate to models/ for the real work)
is what makes rag_nodes easy to test independently of the graph —
you can unit test generate_answer() directly without spinning up
the whole LangGraph workflow.
"""

from src.graph.state import RAGState
from src.models.reasoning_llm import generate_answer


def reasoning_node(state: RAGState) -> dict:
    """
    Generates the final answer using the assembled context.
    """
    query = state["query"]
    context = state["final_context"]

    if not context.strip():
        # Shouldn't normally happen (retrieval always returns something
        # unless the FAISS index is empty), but fail loudly rather than
        # sending an empty-context prompt to the LLM and getting a
        # confused/hallucinated answer back.
        print("WARNING: final_context is empty — check retrieval/compression upstream")

    answer = generate_answer(query, context)

    print(f"Reasoning: generated answer ({len(answer)} characters)")

    return {"answer": answer}