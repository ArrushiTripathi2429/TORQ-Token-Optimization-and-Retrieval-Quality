"""
src/graph/nodes/budget_check_node.py

Node 2 of the TORQ graph: Budget Check.

Counts total tokens across retrieved_chunks and compares against
state["token_budget"]. This node itself doesn't branch — it just
computes current_token_count and puts it in state. The actual
routing decision (go to compression vs skip straight to assembly)
happens in edges.py's route_after_budget_check(), which reads the
values this node produces.
"""

from src.graph.state import RAGState
from src.utils.token_counter import count_tokens_for_chunks


def budget_check_node(state: RAGState) -> dict:
    """
    Computes how many tokens the retrieved chunks add up to.
    Does not modify retrieved_chunks — just measures them.
    """
    retrieved_chunks = state["retrieved_chunks"]
    token_count = count_tokens_for_chunks(retrieved_chunks)
    budget = state["token_budget"]

    over_budget = token_count > budget
    print(f"Budget check: {token_count} tokens retrieved vs {budget} token budget "
          f"({'OVER — compression needed' if over_budget else 'within budget'})")

    return {
        "current_token_count": token_count,
    }