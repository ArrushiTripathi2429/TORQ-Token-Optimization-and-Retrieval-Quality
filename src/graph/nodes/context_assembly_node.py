"""
src/graph/nodes/context_assembly_node.py

Node 4 of the TORQ graph: Context Assembly.

By the time this node runs, one of two things is true:
- Compression ran -> state["compressed_chunks"] has the budget-fitted selection
- Compression was skipped (within budget) -> state["compressed_chunks"] is
  empty/unset, and state["retrieved_chunks"] is what we should use as-is

This node's job is just to pick whichever is correct and format it into
a single clean string the reasoning LLM can read, with source
attribution per chunk (useful for the reasoning model to cite where
facts came from, and for debugging which document an answer traces to).
"""

from src.graph.state import RAGState


def context_assembly_node(state: RAGState) -> dict:
    """
    Selects the correct chunk set (compressed if compression ran,
    otherwise the original retrieval) and formats it into final_context.
    """
    if state.get("compression_applied") and state.get("compressed_chunks"):
        chunks_to_use = state["compressed_chunks"]
        source = "compressed"
    else:
        chunks_to_use = state["retrieved_chunks"]
        source = "uncompressed (within budget)"

    formatted_sections = []
    for chunk in chunks_to_use:
        source_doc = chunk.get("source_doc", "unknown source")
        text = chunk.get("text", "")
        formatted_sections.append(f"[Source: {source_doc}]\n{text}")

    final_context = "\n\n---\n\n".join(formatted_sections)

    print(f"Context assembly: {len(chunks_to_use)} chunks used ({source}), "
          f"final context length: {len(final_context)} characters")

    return {"final_context": final_context}