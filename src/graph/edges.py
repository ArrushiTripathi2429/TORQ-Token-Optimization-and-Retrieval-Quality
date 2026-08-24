"""
src/graph/edges.py

Conditional routing functions for the TORQ graph. Nodes only compute
and update state — they never decide "what happens next." That
decision lives here, in functions that inspect state and return a
string key telling the graph which node to go to.

Two decision points in TORQ:
1. route_after_budget_check — compression needed or not?
2. route_after_quality      — accept the answer, or retry?

These functions are registered with graph.add_conditional_edges() in
build_graph.py, along with a mapping dict from each possible return
string to the actual next node.
"""

from src.graph.state import RAGState

# Threshold below which an answer is considered low quality and worth
# retrying (if retries remain). Move to config.py once that exists.
QUALITY_THRESHOLD = 0.7


def route_after_budget_check(state: RAGState) -> str:
    """
    Decision point 1: are retrieved_chunks within budget?

    - Within budget -> skip compression, go straight to context assembly
    - Over budget   -> go to compression node to select/summarize down

    Returns a string key. The mapping from these keys to actual node
    names is defined in build_graph.py's add_conditional_edges call.
    """
    if state["current_token_count"] <= state["token_budget"]:
        print("Routing: within budget -> context_assembly")
        return "within_budget"
    else:
        print("Routing: over budget -> compression")
        return "over_budget"


def route_after_quality(state: RAGState) -> str:
    """
    Decision point 2: is the answer good enough, or should we retry?

    Three possible outcomes:
    - Quality passed -> proceed to metrics (done)
    - Quality failed but retries remain -> loop back to compression
      (retry with, e.g., a different chunk selection or looser budget)
    - Quality failed and retries exhausted -> proceed to metrics anyway,
      accepting the best-effort answer rather than looping forever

    This third branch is what prevents infinite loops in combination
    with the recursion_limit set at graph invocation time — two
    independent safety nets, not just one.
    """
    quality_passed = state["quality_score"] >= QUALITY_THRESHOLD

    if quality_passed:
        print(f"Routing: quality {state['quality_score']:.2f} passed -> metrics")
        return "passed"

    if state["retry_count"] >= state["max_retries"]:
        print(f"Routing: quality {state['quality_score']:.2f} failed, "
              f"but retries exhausted ({state['retry_count']}/{state['max_retries']}) -> metrics (best-effort)")
        return "retries_exhausted"

    print(f"Routing: quality {state['quality_score']:.2f} failed, "
          f"retrying ({state['retry_count'] + 1}/{state['max_retries']}) -> compression")
    return "retry"


def increment_retry_count(state: RAGState) -> dict:
    """
    Small helper node/update — call this when looping back for a retry,
    so retry_count actually advances and route_after_quality eventually
    hits retries_exhausted instead of looping forever regardless of the
    recursion_limit safety net.
    """
    return {"retry_count": state["retry_count"] + 1}