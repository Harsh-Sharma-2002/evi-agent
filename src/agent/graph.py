import numpy as np
from typing import Callable

from .state import AgentState
from .decisions import decision_node, should_cache
from ..retrieval.cache import VectorCache
from ..retrieval.context_builder import context_expansion_node
from ..retrieval.pubmed import pubmed_fetch_node
from ..scoring.score import score_node


# -------------------------------------------------
# Tier 1: Query cache lookup node (defined ONLY here)
# -------------------------------------------------

def query_cache_node(state: AgentState, cache: VectorCache) -> AgentState:
    """
    Tier 1: Query cache lookup.
    Populates cache_hit / cache_payload / final_answer (if present).
    Does NOT set decision or stop_reason.
    """
    result = cache.search_query(state["query_embedding"])
    if result is None:
        state["cache_hit"] = False
        state["cache_payload"] = None
        return state

    _, payload, _ = result
    state["cache_hit"] = True
    state["cache_payload"] = payload

    if "answer" in payload:
        state["final_answer"] = payload["answer"]

    return state


# -------------------------------------------------
# Tier 2: Chunk store search node
# -------------------------------------------------

def chunk_store_search_node(state: AgentState, cache: VectorCache) -> AgentState:
    """
    Tier 2: Retrieve anchor chunks from chunk store.
    """
    state["anchor_chunks"] = cache.search_chunks(query_embedding=state["query_embedding"])
    return state


# -------------------------------------------------
# Main runner (manual loop, graph-style)
# -------------------------------------------------

def run_agent(
    query: str,
    embed: Callable[[str], list[float]],
    cache: VectorCache,
) -> AgentState:
    """
    3-tier control flow (no LangGraph wiring yet).

    Tier 1: query cache
    Tier 2: chunk store reuse (+ scoring + decision)
    Tier 3: PubMed fetch (fallback) → loops back to Tier 2
    """

    # ---- fresh state ----
    q_emb = np.asarray(embed(query))  #  invariant enforced at boundary

    state: AgentState = {
        "query": query,
        "query_embedding": q_emb,

        "iteration": 0,
        "api_calls": 0,
        "decision": None,
        "stop_reason": None,
        "prev_retrieval_scores": [],

        "documents": [],
        "doc_chunks_map": {},

        "anchor_chunks": [],

        "retrieval_score": 0.0,
        "num_docs": 0,
        "doc_scores": {},
        "confident": False,

        "expanded_context": None,
        "final_answer": None,

        "cache_hit": False,
        "cache_payload": None,
    }

    # ---- Tier 1: query cache ----
    state = query_cache_node(state, cache)
    if state["cache_hit"]:
        return state

    # ---- main loop (Tier 2 ↔ Tier 3) ----
    while True:
        # Tier 2: reuse chunk store
        state = chunk_store_search_node(state, cache)
        state = score_node(state)
        state = decision_node(state)

        if state["decision"] == "STOP":
            break

        # FETCH_MORE edge → Tier 3 PubMed fetch
        state = pubmed_fetch_node(state, cache, embed=embed, retmax=5)

        #  iteration must advance for pagination/offset
        state["iteration"] += 1

    # ---- STOP path: context expansion ----
    if state["anchor_chunks"]:
        state = context_expansion_node(state, window_size=1)

    # NOTE: LLM synthesis node should set state["final_answer"] here.

    # ---- Cache admission ----
    if should_cache(state):
        cache.add_query(
            query=state["query"],
            embedding=state["query_embedding"],
            payload={
                "answer": state["final_answer"],
                "doc_scores": state["doc_scores"],
                "retrieval_score": state["retrieval_score"],
                "num_docs": state["num_docs"],
            },
        )

    return state
