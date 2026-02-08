"""
Entry point for the agentic evidence retrieval system.

- Initializes persistent memory (cache + chat memory)
- Runs an interactive chat loop
- AgentState is created INSIDE the graph per query
"""

from agent.graph import build_agent_graph
from retrieval.cache import VectorCache
from utils.memory import ChatMemory
from utils.logging import log_agent_run


def main():
    # ---------------------------------------------
    # Persistent cross-query memory
    # ---------------------------------------------
    cache = VectorCache()
    memory = ChatMemory()

    # Build LangGraph once
    graph = build_agent_graph(cache, memory)

    print("\n🔬 Agent ready. Type 'exit' to quit.\n")

    # ---------------------------------------------
    # Chat loop
    # ---------------------------------------------
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query or query.lower() == "exit":
            print("Goodbye.")
            break

        # Run agent (AgentState created inside graph)
        state = graph.invoke({"query": query})

        log_agent_run(state)

        print("\nAgent:", state.get("final_answer", ""))
        print(
            f"(stop_reason={state.get('stop_reason')}, "
            f"api_calls={state.get('api_calls')}, "
            f"cache_hit={state.get('cache_hit')}, "
            f"query_cache_size={state.get('query_cache_size')}, "
            f"chunk_store_size={state.get('chunk_store_size')}, "
            f"anchor_chunks={state.get('num_anchor_chunks')})\n"
        )



if __name__ == "__main__":
    main()
