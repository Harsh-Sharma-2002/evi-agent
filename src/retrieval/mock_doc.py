from typing import List, Dict, Any
from agent.state import AgentState
from retrieval.cache import VectorCache
from utils.llm import embed
import numpy as np


MOCK_DOCS = [
    {
        "pmid": "D1",
        "year": 2021,
        "text": (
            "Glioblastoma is an aggressive primary brain tumor. "
            "MRI features such as contrast enhancement and necrosis "
            "are associated with poor prognosis."
        ),
    },
    {
        "pmid": "D2",
        "year": 2020,
        "text": (
            "Radiomic features extracted from MRI scans can be used "
            "to predict survival in patients with glioblastoma."
        ),
    },
    {
        "pmid": "D3",
        "year": 2019,
        "text": (
            "Prognosis of brain tumors depends on tumor type, grade, "
            "and molecular markers in addition to imaging features."
        ),
    },
]


def mock_fetch_node(state: AgentState, cache: VectorCache) -> AgentState:
    state["api_calls"] += 1

    chunk_texts = []
    chunk_embeddings = []
    chunk_metadatas = []

    for i, doc in enumerate(MOCK_DOCS):
        text = doc["text"]
        emb = np.asarray(embed(text))

        chunk_texts.append(text)
        chunk_embeddings.append(emb)
        chunk_metadatas.append(
            {
                "pmid": doc["pmid"],
                "year": doc["year"],
                "chunk_index": 0,
                "section": "mock",
            }
        )

        state["doc_chunks_map"].setdefault(doc["pmid"], []).append(
            {
                "text": text,
                "chunk_index": 0,
                "metadata": chunk_metadatas[-1],
            }
        )

    cache.add_chunks(
        chunks=chunk_texts,
        embeddings=chunk_embeddings,
        metadatas=chunk_metadatas,
    )

    state["documents"].extend(MOCK_DOCS)
    return state
