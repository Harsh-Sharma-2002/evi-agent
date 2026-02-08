from typing import Literal
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from retrieval.cache import VectorCache
#from retrieval.mock_doc import mock_fetch_node as pubmed_fetch_node
from retrieval.context_builder import context_expansion_node
from utils.memory import ChatMemory
from agent.nodes import init_state_node, query_cache_node, cache_write_node, chunk_store_search_node, pubmed_node, prompt_node, score_node, decision_node





# =================================================
# Routing (Edges)
# =================================================

def route_after_cache(state: AgentState) -> Literal["end", "retrieve"]:
    return "prompt" if state["cache_hit"] else "retrieve"


def route_after_decision(state: AgentState) -> Literal["stop", "fetch"]:
    return "stop" if state["decision"] == "STOP" else "fetch"


# Graph Builder


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
            "prompt": "prompt",
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


# Chat Loop (persistent cache + memory)


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
