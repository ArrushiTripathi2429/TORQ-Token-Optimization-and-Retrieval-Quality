"""
main.py

Entry point for TORQ. Builds the initial state for a query, runs it
through the compiled graph, and prints the result along with the
metrics collected along the way.

This is the file you actually run from the terminal:

    python main.py --query "What is the funding structure under PMGSY Phase III?"
    python main.py --query "..." --budget 3000
    python main.py --query "..." --stream   (see each node's output live)

Loads GROQ_API_KEY from .env — make sure that file exists before
running (see .env.example if you created one).
"""

import argparse
import json
from dotenv import load_dotenv

load_dotenv()  # must run before any module that reads GOOGLE_API_KEY is imported/used

from src.graph.build_graph import app
from src.graph.state import get_initial_state


def run_query(query: str, token_budget: int = 2000, max_retries: int = 2, stream: bool = False):
    """
    Runs a single query through the full TORQ graph.

    config={"recursion_limit": ...} is the graph-level safety net for
    the retry cycle — independent of the retry_count check inside
    route_after_quality. Two separate safeguards, not one.
    """
    initial_state = get_initial_state(query=query, token_budget=token_budget, max_retries=max_retries)
    config = {"recursion_limit": 25}

    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"TOKEN BUDGET: {token_budget}")
    print(f"{'='*60}\n")

    if stream:
        # Streams state after every node — useful for demo.ipynb to
        # show the pipeline working step by step instead of a black box.
        final_state = None
        for step_output in app.stream(initial_state, config=config):
            node_name = list(step_output.keys())[0]
            print(f"--- Node completed: {node_name} ---")
            final_state = step_output[node_name]
        result = final_state
    else:
        result = app.invoke(initial_state, config=config)

    print(f"\n{'='*60}")
    print("ANSWER:")
    print(result["answer"])
    print(f"\n{'='*60}")
    print("METRICS:")
    print(json.dumps(result["metrics_log"], indent=2, default=str))
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a query through the TORQ pipeline")
    parser.add_argument("--query", type=str, required=True, help="The question to ask")
    parser.add_argument("--budget", type=int, default=2000, help="Token budget for context")
    parser.add_argument("--max_retries", type=int, default=2, help="Max quality-eval retries")
    parser.add_argument("--stream", action="store_true", help="Stream node-by-node output")
    args = parser.parse_args()

    run_query(
        query=args.query,
        token_budget=args.budget,
        max_retries=args.max_retries,
        stream=args.stream,
    )