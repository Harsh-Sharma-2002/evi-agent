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
from scoring.score import recency_score, similarity_score, doc_diversity_bonus
from agent.decisions import is_stagnating




# Scoring hyperparameters
RECENCY_HALFLIFE_YEARS = 8          # publication age decay
MIN_DOCS_FOR_CONFIDENCE = 2         # hard confidence gate
MAX_SCORE = 1.0                    # upper bound for retrieval score

# Decisions hyperparameters 
STOP_SCORE_THRESHOLD = 0.6 # Min confidence to stop
MIN_CACHE_SCORE = 0.6 # Min quality to reuse later ie enter in cache  

MAX_ITERATIONS = 4 # prevent excess looping
MAX_API_CALLS = 5 # API call control else it would keep calling and exceed rate limit

STAGNATION_TOLERATION = 0.02 # If after calls no better result then stop and say I dont know man

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

    - Consumes state["anchor_chunks"] (from Tier 2 or Tier 3)
    - This node is TIER-AGNOSTIC :- it does not care where the chunks came from.
    """

    anchor_chunks = state["anchor_chunks"]


    # No evidence case (valid outcome, not an error)
    if not anchor_chunks:
        state["retrieval_score"] = 0.0
        state["num_docs"] = 0
        state["doc_scores"] = {}
        state["confident"] = False

        # Track score evolution for stagnation detection
        state["prev_retrieval_scores"].append(0.0)
        return state

    # Accumulate chunk-level scores per document
    doc_scores_accumulator: Dict[str, List[float]] = defaultdict(list)

    for chunk in anchor_chunks:
        meta = chunk.get("metadata", {})
        pmid = meta.get("pmid")

        if pmid is None:
            continue

        sim = similarity_score(chunk.get("similarity", 0.0))
        year = meta.get("year")

        chunk_score = (
            0.7 * sim +
            0.3 * recency_score(year)
        )

        doc_scores_accumulator[pmid].append(chunk_score)


    # Collapse chunks → per-document score (max)
    doc_scores: Dict[str, float] = {
        pmid: max(scores)
        for pmid, scores in doc_scores_accumulator.items()
    }

    num_docs = len(doc_scores)

    if num_docs == 0:
        state["retrieval_score"] = 0.0
        state["num_docs"] = 0
        state["doc_scores"] = {}
        state["confident"] = False

        state["prev_retrieval_scores"].append(0.0)
        return state

    # Aggregate document-level evidence
    avg_doc_score = sum(doc_scores.values()) / num_docs
    diversity = doc_diversity_bonus(num_docs)

    retrieval_score = min(
        MAX_SCORE,
        avg_doc_score * diversity
    )

    confident = num_docs >= MIN_DOCS_FOR_CONFIDENCE

    # Update agent state
    state["retrieval_score"] = retrieval_score
    state["num_docs"] = num_docs
    state["doc_scores"] = doc_scores
    state["confident"] = confident

    # Track score evolution for stagnation detection
    state["prev_retrieval_scores"].append(retrieval_score)

    return state
#####################################################################################

def decision_node(state: AgentState) -> AgentState:
    """
    - LangGraph node.
    - Decides whether the agent should STOP or FETCH_MORE
    based on current state, and records the reason in
    state["stop_reason"].
    - This node is Tier-agnostic, stateless beyond AgentState
    """
     
    # Tier 1: Cache hit → immediate stop

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

