from  .state import AgentState
from typing import Literal,List


STOP_SCORE_THRESHOLD = 0.6 # Min confidence to stop
MIN_CACHE_SCORE = 0.6 # Min quality to reuse later ie enter in cache  

MAX_ITERATIONS = 4 # prevent excess looping
MAX_API_CALLS = 5 # API call control else it would keep calling and exceed rate limit

STAGNATION_TOLERATION = 0.02 # If after calls no better result then stop and say I dont know man


def is_stagnating(scores: List[float],tol: float = STAGNATION_TOLERATION) -> bool:
    """
    - Detect whether retrieval score is no longer improving.
    - Used to stop retrieval when additional fetches
      are unlikely to yield better evidence.
    """
    if len(scores) < 2:
        return False
    
    return abs(scores[-1] - scores[-2]) < tol


def decision_node(state: AgentState) -> AgentState:
    """
    - LangGraph node.
    - Decides whether the agent should STOP or FETCH_MORE
    based on current state, and records the reason in
    state["stop_reason"].
    - This node is Tier-agnostic, stateless beyond AgentState
    """
     
    # Tier 1: Cache hit → immediate stop
    if state["cache_hit"]:
        state["decision"] = "STOP"
        state["stop_reason"] = "cache_hit"
        return state

    # Hard safety limits (override everything)
    if state["iteration"] >= MAX_ITERATIONS:
        state["decision"] = "STOP"
        state["stop_reason"] = "max_iterations"
        return state

    if state["api_calls"] >= MAX_API_CALLS:
        state["decision"] = "STOP"
        state["stop_reason"] = "api_limit"
        return state

    # No evidence and exhausted
    if state["num_docs"] == 0 and state["evidence_exhausted"]:
        state["decision"] = "STOP"
        state["stop_reason"] = "no_evidence"
        return state
    
    # No evidence yet → must fetch more
    if state["num_docs"] == 0:
        state["decision"] = "FETCH_MORE"
        state["stop_reason"] = None
        return state


    # Stagnation detection 
    if is_stagnating(state["prev_retrieval_scores"]):
        state["decision"] = "STOP"
        state["stop_reason"] = "stagnation"
        return state

    # Quality-based stopping rule (happy path)
    if (
        state["retrieval_score"] >= STOP_SCORE_THRESHOLD
        and state["confident"]
    ):
        state["decision"] = "STOP"
        state["stop_reason"] = "score_threshold_met"
        return state

    # Otherwise → fetch more evidence
    state["decision"] = "FETCH_MORE"
    state["stop_reason"] = None
    return state


def should_cache(state: AgentState) -> bool:
    """
    - Decide whether the final result should be stored
    in the query cache.
    - Only confident, high-quality, non-forced stops
    are admitted.
    """
    return ( state["decision"] == "STOP" and state["stop_reason"] == "score_threshold_met" and state["api_calls"] > 0 and state["retrieval_score"] >= MIN_CACHE_SCORE and state["num_docs"] >= 1 )