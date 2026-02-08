import time
from pathlib import Path
from agent.state import AgentState

LOG_FILE = Path("logs.txt")


def log_agent_run(state: AgentState) -> None:
    """
    Persist a completed agent run to disk.

    This logger is PURE observability:
    - Reads counters already recorded in AgentState
    - Does NOT infer or recompute metrics
    - Does NOT mutate state
    """

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"""
==============================
Timestamp: {timestamp}

Query:
{state.get("query")}

Final Answer:
{state.get("final_answer")}

Stop Reason: {state.get("stop_reason")}

--- Reuse Accounting ---
Query Cache Hits (Tier-1): {state.get("query_cache_hits")}
Chunk Store Hits (Tier-2): {state.get("chunk_store_hits")}
Total Reuse Events: {state.get("query_cache_hits") + state.get("chunk_store_hits")}

--- API Usage ---
API Calls Used: {state.get("api_calls")}
API Calls Saved: {state.get("api_calls_saved")}

--- Memory State ---
Query Cache Size: {state.get("query_cache_size")}
Chunk Store Size: {state.get("chunk_store_size")}
Anchor Chunks Selected: {state.get("num_anchor_chunks")}

--- Retrieval Quality ---
Retrieval Score: {state.get("retrieval_score")}
Number of Documents: {state.get("num_docs")}
==============================

"""

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)
