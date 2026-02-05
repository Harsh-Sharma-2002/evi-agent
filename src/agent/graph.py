import numpy as np
from typing import Literal

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.decisions import decision_node, should_cache

from retrieval.cache import VectorCache
from retrieval.pubmed import pubmed_fetch_node
from retrieval.context_builder import context_expansion_node
from scoring.score import score_node

from utils.llm import embed, call_llm
from utils.prompt import build_final_prompt
from utils.memory import ChatMemory


# =================================================
# Node functions
# =================================================

def init_state_node(query: str) -> AgentState:
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
    }


def query_cache_node(state: AgentState, cache: VectorCache) -> AgentState:
    result = cache.search_query(state["query_embedding"])
    if result is None:
        state["cache_hit"] = False
        return state

    _, payload, _ = result
    state["cache_hit"] = True
    state["cache_payload"] = payload
    state["final_answer"] = payload.get("answer")
    return state


def chunk_store_search_node(state: AgentState, cache: VectorCache) -> AgentState:
    state["anchor_chunks"] = cache.search_chunks(
        query_embedding=state["query_embedding"]
    )
    return state


def pubmed_node(state: AgentState, cache: VectorCache) -> AgentState:
    state = pubmed_fetch_node(state, cache)
    state["iteration"] += 1
    return state


def prompt_node(state: AgentState, memory: ChatMemory) -> AgentState:
    prompt = build_final_prompt(
        state,
        chat_memory=memory.get_memory_context(),
    )
    state["final_answer"] = call_llm(state, prompt)
    memory.update(state)
    return state


def cache_write_node(state: AgentState, cache: VectorCache) -> AgentState:
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


# =================================================
# Routing (conditional edges)
# =================================================

def route_after_cache(state: AgentState) -> Literal["end", "retrieve"]:
    return "end" if state["cache_hit"] else "retrieve"


def route_after_decision(state: AgentState) -> Literal["stop", "fetch"]:
    return "stop" if state["decision"] == "STOP" else "fetch"


# =================================================
# Graph builder
# =================================================

def build_agent_graph(cache: VectorCache, memory: ChatMemory):
    g = StateGraph(AgentState)

    # ---- add nodes ----
    g.add_node("cache", lambda s: query_cache_node(s, cache))
    g.add_node("retrieve", lambda s: chunk_store_search_node(s, cache))
    g.add_node("score", score_node)
    g.add_node("decide", decision_node)
    g.add_node("fetch", lambda s: pubmed_node(s, cache))
    g.add_node("context", context_expansion_node)
    g.add_node("prompt", lambda s: prompt_node(s, memory))
    g.add_node("cache_write", lambda s: cache_write_node(s, cache))

    # ---- entry ----
    g.set_entry_point("cache")

    # ---- Tier 1 routing ----
    g.add_conditional_edges(
        "cache",
        route_after_cache,
        {
            "end": END,
            "retrieve": "retrieve",
        },
    )

    # ---- Tier 2 path ----
    g.add_edge("retrieve", "score")
    g.add_edge("score", "decide")

    # ---- decision routing ----
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
# Public runner
# =================================================

def run_agent(query: str, cache: VectorCache, memory: ChatMemory) -> AgentState:
    graph = build_agent_graph(cache, memory)
    state = init_state_node(query)
    return graph.invoke(state)
