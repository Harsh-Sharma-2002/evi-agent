### Add the the system prompt to add the results and chunks in output and then use the chunks and resutl in the query cache
from .state import AgentState
from ..retrieval.cache import VectorCache
"""
You are a research assistant.

Given the provided context, return your response in the following JSON format ONLY:

{
  "answer": "<clear, concise answer>",
  "supporting_points": [
    "<short factual statement>",
    "<short factual statement>"
  ]
}

Do not include citations, explanations, or extra text outside this JSON.

"""
graph = None
embed = None

def run_agent(query: str, cache: VectorCache):
    # 🔄 NEW STATE EVERY TIME
    state: AgentState = {
        "query": query,
        "query_embedding": embed(query),

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

    # ▶ Run graph ONCE
    final_state = graph.invoke(state)

    return final_state["final_answer"]
