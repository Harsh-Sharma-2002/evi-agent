from dataclasses import dataclass
from typing import Optional

from agent.state import AgentState
from utils.llm import call_llm


@dataclass
class ChatMemory:
    """
    Summary-only conversational memory.

    - Stores ONE rolling summary string
    - Uses LLM only when update threshold is crossed
    - Designed to complement query LRU cache
    """

    summarize_every: int = 10  # summarize after N interactions
    summary: str = ""
    _interaction_count: int = 0

    # Update memory

    def update(self, state: AgentState) -> None:
        """
        Update memory based on the latest interaction.

        This should be called AFTER the agent produces an answer.
        """

        self._interaction_count += 1

        if self._interaction_count < self.summarize_every:
            return  

        self._interaction_count = 0
        self._summarize(state)

    # Summarization

    def _summarize(self, state: AgentState) -> None:
        """
        Compress conversation state into a concise memory summary.
        """

        query = state["query"]
        answer = state.get("final_answer") or ""

        prompt = f"""
You are maintaining long-term conversational memory.

Summarize the interaction below into a concise factual memory.
Preserve:
- user goals and intent
- constraints and preferences
- conclusions reached
- unresolved questions

Do NOT include chit-chat.
Do NOT include implementation details.

Existing memory:
{self.summary or "None"}

New interaction:
User question:
{query}

Assistant answer:
{answer}

Updated memory:
""".strip()

        new_summary = call_llm(
            state=state,
            prompt=prompt,
            max_new_tokens=200,
        )

        if new_summary:
            self.summary = new_summary.strip()

  
    # Memory access

    def get_memory_context(self) -> Optional[str]:
        """
        Returns memory summary for prompt injection.
        """
        return self.summary if self.summary else None
