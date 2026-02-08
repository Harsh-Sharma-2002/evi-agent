from  .state import AgentState
from typing import List, Dict, Any, Optional


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




def should_cache(state: AgentState) -> bool:
    """
    - Decide whether the final result should be stored
    in the query cache.
    - Only confident, high-quality, non-forced stops
    are admitted.
    """
    return ( state["decision"] == "STOP" and state["stop_reason"] == "score_threshold_met" and state["api_calls"] > 0 and state["retrieval_score"] >= MIN_CACHE_SCORE and state["num_docs"] >= 1 )
    

def decide_action(state: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    PURE decision policy.

    Input: read-only snapshot of agent state
    Output:
        {
            "decision": "STOP" | "FETCH_MORE",
            "stop_reason": Optional[str]
        }
    """

    # Hard safety limits
    if state["iteration"] >= MAX_ITERATIONS:
        return {"decision": "STOP", "stop_reason": "max_iterations"}

    if state["api_calls"] >= MAX_API_CALLS:
        return {"decision": "STOP", "stop_reason": "api_limit"}

    # No evidence and exhausted
    if state["num_docs"] == 0 and state["evidence_exhausted"]:
        return {"decision": "STOP", "stop_reason": "no_evidence"}

    # No evidence yet → must fetch more
    if state["num_docs"] == 0:
        return {"decision": "FETCH_MORE", "stop_reason": None}

    # Stagnation detection
    if is_stagnating(state["prev_retrieval_scores"]):
        return {"decision": "STOP", "stop_reason": "stagnation"}

    # Quality-based stopping rule (happy path)
    if state["retrieval_score"] >= STOP_SCORE_THRESHOLD and state["confident"]:
        return {"decision": "STOP", "stop_reason": "score_threshold_met"}

    # Otherwise → fetch more evidence
    return {"decision": "FETCH_MORE", "stop_reason": None}
