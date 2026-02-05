from typing import Optional
from agent.state import AgentState
from retrieval.context_builder import (
    context_expansion_node,
    flatten_context_from_state,
)


def build_final_prompt(state: AgentState, chat_memory: Optional[str] = None, window_size: int = 1,) -> str:
    """
    Build the FINAL prompt for the LLM.

    Responsibilities:
    - Run context window expansion
    - Read expanded context
    - Incorporate optional chat memory
    - Produce a single grounded prompt string
    """


    # Ensure context is expanded (idempotent)
    state = context_expansion_node(state, window_size=window_size)

    query = state["query"]
    context_chunks = flatten_context_from_state(state)

    # Base system instructions

    system_rules = """
You are a scientific research assistant.

Rules:
- Answer ONLY using the provided evidence.
- Do NOT use prior knowledge.
- Do NOT hallucinate.
- If evidence is insufficient or conflicting, say so clearly.
- Be concise and factual.
""".strip()

    # -------------------------------------------------
    # Optional conversational memory
    # -------------------------------------------------
    memory_block = ""
    if chat_memory:
        memory_block = f"""
Conversation Memory:
{chat_memory}
""".strip()

    # -------------------------------------------------
    # Evidence block
    # -------------------------------------------------
    if context_chunks:
        evidence_block = "\n\n".join(
            f"[Evidence {i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        )
    else:
        evidence_block = (
            "No relevant evidence was retrieved for this question."
        )

    # -------------------------------------------------
    # Final prompt assembly
    # -------------------------------------------------
    prompt = f"""
{system_rules}

{memory_block}

Question:
{query}

Evidence:
{evidence_block}

Answer:
""".strip()

    return prompt
