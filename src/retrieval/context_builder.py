from typing import Dict, List, Any
from collections import defaultdict

from agent.state import AgentState



# Context window expansion (LangGraph node)

def context_expansion_node(state: AgentState, window_size: int = 1) -> AgentState:
    """
    LangGraph node.

    - Expands context windows around anchor chunks and
      stores the result in state["expanded_context"].
    - Called only after the agent decides STOP.
    """

    anchor_chunks = state["anchor_chunks"]
    doc_chunks_map = state["doc_chunks_map"]

    # Defensive: nothing to expand
    if not anchor_chunks or not doc_chunks_map:
        state["expanded_context"] = {}
        return state

    # pmid -> chunk_index -> chunk
    expanded: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)

    for anchor in anchor_chunks:
        meta = anchor.get("metadata", {})
        pmid = meta.get("pmid")
        anchor_idx = meta.get("chunk_index")

        if pmid is None or anchor_idx is None:
            continue

        if pmid not in doc_chunks_map:
            continue

        # Ensure chunks are ordered by chunk_index
        all_chunks = sorted(doc_chunks_map[pmid], key=lambda c: c.get("chunk_index", 0))

        total = len(all_chunks)

        # Anchor position in ordered list
        try:
            anchor_pos = next(i for i, c in enumerate(all_chunks) if c.get("chunk_index") == anchor_idx)
        except StopIteration:
            continue

        start = max(0, anchor_pos - window_size)
        end = min(total, anchor_pos + window_size + 1)

        for i in range(start, end):
            chunk = all_chunks[i]
            idx = chunk.get("chunk_index")
            if idx is not None:
                expanded[pmid][idx] = chunk

    # Final ordered context per document
    final_context: Dict[str, List[Dict[str, Any]]] = {}

    for pmid, chunks_by_idx in expanded.items():
        final_context[pmid] = [
            chunks_by_idx[idx]
            for idx in sorted(chunks_by_idx.keys())
        ]

    state["expanded_context"] = final_context
    return state


# Utility: flatten context for LLM prompt

def flatten_context_from_state(state: AgentState) -> List[str]:
    """
    Convert expanded_context in AgentState into a flat,
    deterministic list of strings suitable for prompt construction.
    """

    expanded_context = state.get("expanded_context")

    if not expanded_context:
        return []

    texts: List[str] = []

    for pmid in sorted(expanded_context.keys()):
        for chunk in expanded_context[pmid]:
            text = chunk.get("text")
            if text:
                texts.append(text)

    return texts

