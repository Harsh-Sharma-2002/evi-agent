from typing import Dict, List, Any
from collections import defaultdict

from agent.state import AgentState


# -------------------------------------------------
# Context window expansion (LangGraph node)
# -------------------------------------------------

def context_expansion_node(
    state: AgentState,
    window_size: int = 1,
) -> AgentState:
    """
    LangGraph node.

    Expands context windows around anchor chunks and
    stores the result in state["expanded_context"].

    This node MUST only be called after the agent decides STOP.
    """

    anchor_chunks = state["anchor_chunks"]
    doc_chunks_map = state["doc_chunks_map"]

    expanded: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)

    for anchor in anchor_chunks:
        meta = anchor.get("metadata", {})
        pmid = meta.get("pmid")
        idx = meta.get("chunk_index")

        if pmid is None or idx is None:
            continue

        if pmid not in doc_chunks_map:
            continue

        all_chunks = doc_chunks_map[pmid]
        total = len(all_chunks)

        start = max(0, idx - window_size)
        end = min(total, idx + window_size + 1)

        for i in range(start, end):
            chunk = all_chunks[i]
            expanded[pmid][chunk["chunk_index"]] = chunk

    # Order chunks per document by chunk_index
    final_context: Dict[str, List[Dict[str, Any]]] = {}

    for pmid, chunks_by_idx in expanded.items():
        ordered = [
            chunks_by_idx[i]
            for i in sorted(chunks_by_idx.keys())
        ]
        final_context[pmid] = ordered

    # Update agent state
    state["expanded_context"] = final_context

    return state


# -------------------------------------------------
# Utility: flatten context for LLM prompt
# -------------------------------------------------

def flatten_context_from_state(state: AgentState) -> List[str]:
    """
    Convert expanded_context in AgentState into a flat
    list of strings suitable for prompt construction.
    """

    expanded_context = state.get("expanded_context")

    if not expanded_context:
        return []

    texts: List[str] = []

    for pmid, chunks in expanded_context.items():
        for chunk in chunks:
            texts.append(chunk["text"])

    return texts
