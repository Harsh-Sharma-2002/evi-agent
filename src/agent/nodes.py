import numpy as np
from agent.state import AgentState
from agent.decisions import should_cache

from retrieval.cache import VectorCache
from retrieval.pubmed import pubmed_fetch_node
#from retrieval.mock_doc import mock_fetch_node as pubmed_fetch_node

from utils.llm import embed, call_llm
from utils.prompt import build_final_prompt
from utils.memory import ChatMemory
from typing import Dict, List
from collections import defaultdict
from scoring.score import compute_retrieval_score
from agent.decisions import decide_action



# =================================================
# Graph Nodes
# =================================================

def init_state_node(state: dict) -> AgentState:
    print("[ENTER NODE] init_state_node")

    query = state["query"]

    return {
        "query": query,
        "query_embedding": np.asarray(embed(query)),

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
        "evidence_exhausted": False,
        "query_cache_size": 0,
        "chunk_store_size": 0,
        "num_anchor_chunks": 0,
        "query_cache_hits": 0,
        "chunk_store_hits": 0,
        "api_calls_saved": 0,
    }

#####################################################################################
def query_cache_node(state: AgentState, cache: VectorCache) -> AgentState:
    print("[ENTER NODE] query_cache_node")

    result = cache.search_query(state["query_embedding"])
    if result is None:
        state["cache_hit"] = False
        return state
    
    # Logging state variables
    state["cache_hit"] = True
    state["query_cache_hits"] += 1
    state["api_calls_saved"] += 1

    # Cache data retrieval  
    _, payload, sim = result
    state["cache_hit"] = True
    state["cache_payload"] = payload
    state["cache_payload"] = payload
    state["cache_similarity"] = sim
    state["query_cache_size"] = cache.query_collection.count()
    return state

#####################################################################################

def chunk_store_search_node(state: AgentState, cache: VectorCache) -> AgentState:
    print("[ENTER NODE] chunk_store_search_node")

    # Check for first retrieval 
    if cache.chunk_collection.count() == 0:
        state["anchor_chunks"] = []
        return state

    chunks = cache.search_chunks(
        query_embedding=state["query_embedding"]
    )
    state["anchor_chunks"] = chunks

    # Tier-2 reuse events
    chunk_hits = len(chunks)
    state["chunk_store_hits"] += chunk_hits

    # Memory diagnostics
    state["num_anchor_chunks"] = len(chunks)
    state["chunk_store_size"] = cache.chunk_collection.count()

    return state


#####################################################################################

def pubmed_node(state: AgentState, cache: VectorCache) -> AgentState:
    print("[ENTER NODE] pubmed_fetch_node")

    state = pubmed_fetch_node(state, cache)
    state["iteration"] += 1  # iteration == number of PubMed fetches
    return state

#####################################################################################

def prompt_node(state: AgentState, memory: ChatMemory) -> AgentState:
    print("[ENTER NODE] prompt_node")

    prompt = build_final_prompt(
        state,
        chat_memory=memory.get_memory_context(),
    )
    state["final_answer"] = call_llm(
        state,
        prompt,
        node_name="prompt_node",
    )
    memory.update(state)
    print(prompt)
    return state

#####################################################################################

def cache_write_node(state: AgentState, cache: VectorCache) -> AgentState:
    print("[ENTER NODE] cache_write_node")

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
        # Cache dianostics
        state["query_cache_size"] = cache.query_collection.count()
    return state

#####################################################################################

def score_node(state: AgentState) -> AgentState:
    """
    LangGraph node.

    - Calls pure scoring helper
    - Mutates AgentState
    - Tracks score evolution for stagnation detection
    """

    result = compute_retrieval_score(state["anchor_chunks"])

    state["retrieval_score"] = result["retrieval_score"]
    state["num_docs"] = result["num_docs"]
    state["doc_scores"] = result["doc_scores"]
    state["confident"] = result["confident"]

    # Track score evolution for stagnation detection
    state["prev_retrieval_scores"].append(
        state["retrieval_score"]
    )

    return state

#####################################################################################



def decision_node(state: AgentState) -> AgentState:
    """
    LangGraph node.

    - Delegates decision policy to pure helper
    - Applies decision to AgentState
    """

    result = decide_action(state)

    state["decision"] = result["decision"]
    state["stop_reason"] = result["stop_reason"]

    return state


