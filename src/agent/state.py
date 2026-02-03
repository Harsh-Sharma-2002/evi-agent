from typing import TypedDict, List, Dict, Any, Optional, Literal
import numpy as np


class AgentState(TypedDict):
    """
    Global mutable state passed between LangGraph nodes.
    This is the ONLY place where agent state is defined.
    """
   
    query: str
    query_embedding: np.ndarray 

    iteration: int # Control / loop bookkeeping
    api_calls: int
    decision: Optional[Literal["STOP", "FETCH_MORE"]]
    
    documents: List[Dict[str, Any]]          # raw PubMed docs (# Retrieval results)
    doc_chunks_map: Dict[str, List[Dict[str, Any]]]  # pmid -> chunks
    
    anchor_chunks: List[Dict[str, Any]] # Chunk-level retrieval

    retrieval_score: float # Scoring outputs
    num_docs: int
    doc_scores: Dict[str, float]
    confident: bool

    expanded_context: Optional[Dict[str, List[Dict[str, Any]]]] # Context window expansion part
    final_answer: Optional[str]

    cache_hit: bool # Cache-related
    cache_payload: Optional[Dict[str, Any]]
