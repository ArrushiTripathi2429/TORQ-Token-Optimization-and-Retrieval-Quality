"""
src/graph/build_graph.py

Assembles all 7 nodes into the actual LangGraph StateGraph: registers
every node, wires the fixed (normal) edges, and wires the two
conditional edges using the routing functions from edges.py.

This is the file that turns individual node functions into an actual
executable graph. Everything built so far (state, nodes, edges) comes
together here.
"""

from langgraph.graph import StateGraph, END

from src.graph.state import RAGState
from src.graph.edges import route_after_budget_check, route_after_quality, increment_retry_count

from src.graph.nodes.retrieval_node import retrieval_node
from src.graph.nodes.budget_check_node import budget_check_node
from src.graph.nodes.compression_node import compression_node
from src.graph.nodes.context_assembly_node import context_assembly_node
from src.graph.nodes.reasoning_node import reasoning_node
from src.graph.nodes.quality_eval_node import quality_eval_node
from src.graph.nodes.metrics_node import metrics_node


def build_graph():
    """
    Builds and compiles the TORQ graph.

    Flow:
        retrieval -> budget_check -> [conditional] -> compression? -> context_assembly
                     -> reasoning -> quality_eval -> [conditional] -> retry? or metrics -> END

    Returns a compiled graph (app) ready for .invoke() or .stream().
    """
    graph = StateGraph(RAGState)

    # --- Register all nodes ---
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("budget_check", budget_check_node)
    graph.add_node("compression", compression_node)
    graph.add_node("context_assembly", context_assembly_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("quality_eval", quality_eval_node)
    graph.add_node("increment_retry", increment_retry_count)  # small helper node for the retry loop
    graph.add_node("metrics", metrics_node)

    # --- Entry point ---
    graph.set_entry_point("retrieval")

    # --- Fixed (normal) edges ---
    graph.add_edge("retrieval", "budget_check")
    graph.add_edge("compression", "context_assembly")
    graph.add_edge("context_assembly", "reasoning")
    graph.add_edge("reasoning", "quality_eval")
    graph.add_edge("increment_retry", "compression")  # retry loops back into compression
    graph.add_edge("metrics", END)

    # --- Conditional edge 1: after budget_check ---
    # route_after_budget_check returns "within_budget" or "over_budget"
    graph.add_conditional_edges(
        "budget_check",
        route_after_budget_check,
        {
            "within_budget": "context_assembly",
            "over_budget": "compression",
        },
    )

    # --- Conditional edge 2: after quality_eval ---
    # route_after_quality returns "passed", "retry", or "retries_exhausted"
    graph.add_conditional_edges(
        "quality_eval",
        route_after_quality,
        {
            "passed": "metrics",
            "retry": "increment_retry",
            "retries_exhausted": "metrics",
        },
    )

    return graph.compile()


# Module-level compiled app — import this directly elsewhere
# (e.g. `from src.graph.build_graph import app`) instead of rebuilding
# the graph every time.
app = build_graph()