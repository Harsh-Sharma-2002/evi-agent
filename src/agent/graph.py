import numpy as np
from typing import Literal

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.decisions import decision_node, should_cache

from retrieval.cache import VectorCache
from retrieval.pubmed import pubmed_fetch_node
#from retrieval.mock_doc import mock_fetch_node as pubmed_fetch_node

from retrieval.context_builder import context_expansion_node
from scoring.score import score_node

from utils.llm import embed, call_llm
from utils.prompt import build_final_prompt
from utils.memory import ChatMemory


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
    _, payload, _ = result
    state["cache_hit"] = True
    state["cache_payload"] = payload
    state["final_answer"] = payload.get("answer")
    state["query_cache_size"] = cache.query_collection.count()
    return state


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

    # Conservative API savings: reuse avoids future fetches
    state["api_calls_saved"] += max(0, chunk_hits - 1)
    # Memory diagnostics
    state["num_anchor_chunks"] = len(chunks)
    state["chunk_store_size"] = cache.chunk_collection.count()

    return state


def pubmed_node(state: AgentState, cache: VectorCache) -> AgentState:
    print("[ENTER NODE] pubmed_fetch_node")

    state = pubmed_fetch_node(state, cache)
    state["iteration"] += 1  # iteration == number of PubMed fetches
    return state


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


# =================================================
# Routing (Edges)
# =================================================

def route_after_cache(state: AgentState) -> Literal["end", "retrieve"]:
    return "end" if state["cache_hit"] else "retrieve"


def route_after_decision(state: AgentState) -> Literal["stop", "fetch"]:
    return "stop" if state["decision"] == "STOP" else "fetch"


# =================================================
# Graph Builder
# =================================================

def build_agent_graph(cache: VectorCache, memory: ChatMemory):
    g = StateGraph(AgentState)

    # ---- nodes ----
    g.add_node("init", init_state_node)
    g.add_node("cache", lambda s: query_cache_node(s, cache))
    g.add_node("retrieve", lambda s: chunk_store_search_node(s, cache))
    g.add_node("score", score_node)
    g.add_node("decide", decision_node)
    g.add_node("fetch", lambda s: pubmed_node(s, cache))
    g.add_node("context", context_expansion_node)
    g.add_node("prompt", lambda s: prompt_node(s, memory))
    g.add_node("cache_write", lambda s: cache_write_node(s, cache))

    # ---- entry ----
    g.set_entry_point("init")

    # ---- init → cache ----
    g.add_edge("init", "cache")

    # ---- Tier 1 routing ----
    g.add_conditional_edges(
        "cache",
        route_after_cache,
        {
            "end": END,
            "retrieve": "retrieve",
        },
    )

    # ---- Tier 2 ----
    g.add_edge("retrieve", "score")
    g.add_edge("score", "decide")

    # ---- Decision routing ----
    g.add_conditional_edges(
        "decide",
        route_after_decision,
        {
            "stop": "context",
            "fetch": "fetch",
        },
    )

    # ---- Tier 3 loop ----
    g.add_edge("fetch", "retrieve")

    # ---- STOP path ----
    g.add_edge("context", "prompt")
    g.add_edge("prompt", "cache_write")
    g.add_edge("cache_write", END)

    return g.compile()


# =================================================
# Chat Loop (persistent cache + memory)
# =================================================

def run_agent_loop():
    """
    Interactive chat loop.

    - VectorCache persists across turns
    - ChatMemory persists across turns
    - AgentState is created inside the graph
    """

    cache = VectorCache()
    memory = ChatMemory()

    graph = build_agent_graph(cache, memory)

    print("🔬 Agent ready. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if not query or query.lower() == "exit":
            break

        state = graph.invoke({"query": query})

        print("\nAgent:", state["final_answer"])
        print(
            f"(stop_reason={state['stop_reason']}, "
            f"api_calls={state['api_calls']}, "
            f"cache_hit={state['cache_hit']})\n"
        )
