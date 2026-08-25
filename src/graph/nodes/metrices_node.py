"""
src/graph/nodes/metrics_node.py

Node 7 of the TORQ graph: Metrics Reporting. The last node — every
path through the graph (compressed or not, retried or not, quality
passed or exhausted) ends up here before the graph terminates.

Compiles a single metrics record for this query: token usage,
compression ratio, retry count, and quality outcome. This record gets
appended to state["metrics_log"] (remember the operator.add reducer
in state.py — this node's return value ADDS to the list rather than
overwriting it, which matters if this node is ever reached more than
once for a query, e.g. in future graph variants).

This is the data that feeds results/cost_analysis.csv,
results/quality_metrics.json etc. via metrics_tracker.py, which
aggregates many of these per-query records across a full evaluation run.
"""

import time
from src.graph.state import RAGState


def metrics_node(state: RAGState) -> dict:
    """
    Compiles a metrics record summarizing this query's run: how many
    tokens were retrieved vs actually used, whether compression ran,
    how many retries it took, and the final quality outcome.
    """
    retrieved_count = len(state["retrieved_chunks"])
    used_count = len(state["compressed_chunks"]) if state.get("compression_applied") else retrieved_count

    # Tokens actually used in final_context vs tokens that were
    # available before budgeting — this delta is the core TORQ result.
    tokens_before_budgeting = state["current_token_count"]
    tokens_after_budgeting = min(state["current_token_count"], state["token_budget"]) \
        if state.get("compression_applied") else state["current_token_count"]

    compression_ratio = (
        1 - (tokens_after_budgeting / tokens_before_budgeting)
        if tokens_before_budgeting > 0 else 0.0
    )

    record = {
        "timestamp": time.time(),
        "query": state["query"],
        "token_budget": state["token_budget"],
        "tokens_before_budgeting": tokens_before_budgeting,
        "tokens_after_budgeting": tokens_after_budgeting,
        "compression_applied": state.get("compression_applied", False),
        "compression_ratio": round(compression_ratio, 3),
        "chunks_retrieved": retrieved_count,
        "chunks_used": used_count,
        "retry_count": state["retry_count"],
        "quality_score": state["quality_score"],
        "quality_passed": state["quality_score"] >= 0.7,
    }

    print(f"Metrics: compression_ratio={record['compression_ratio']}, "
          f"quality={record['quality_score']:.2f}, retries={record['retry_count']}")

    # This list gets APPENDED to state["metrics_log"] thanks to the
    # operator.add reducer defined in state.py — not overwritten.
    return {"metrics_log": [record]}