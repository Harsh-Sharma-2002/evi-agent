import time
import uuid
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings


"""
Hybrid vector memory (policy-free).

Tier 1: Query cache
- One vector per solved query                          ### Why is cache policy free, policy vs mechanism, why context expansion after stop, why monotonic similarity matters, 
- Payload-based reuse
- TTL + LRU eviction

Tier 2: Chunk store
- Many vectors per document
- Semantic precision
- Diversity-controlled retrieval
"""


class VectorCache:
    """
    Hybrid vector memory:
    1. Query cache (coarse-grained, LRU + TTL)
    2. Chunk store (fine-grained, semantic precision)

    IMPORTANT:
    - This class does NOT decide what should be cached.
    - The agent decides admission.
    - This class only stores and retrieves.
    """

    def __init__(self,similarity_threshold: float = 0.82, max_query_items: int = 200, ttl_seconds: int = 48 * 3600, max_chunks_per_doc: int = 1,):
        self.similarity_threshold = similarity_threshold
        self.max_query_items = max_query_items
        self.ttl_seconds = ttl_seconds
        self.max_chunks_per_doc = max_chunks_per_doc

        self.client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                is_persistent=False,
            )
        )

        self.query_collection = self.client.get_or_create_collection(
            name="query_cache"
        )
        self.chunk_collection = self.client.get_or_create_collection(
            name="chunk_store"
        )

        # LRU applies ONLY to query cache
        self._lru: OrderedDict[str, float] = OrderedDict()

    # Tier 1: QUERY CACHE

    def search_query(self, query_embedding: np.ndarray, max_candidates: int = 3,) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """
        - Search for a reusable cached query.
        - Returns:
            (cache_id, payload, similarity)
        - payload is the reusable unit.
         similarity is returned for diagnostics only.
        """

        if self.query_collection.count() == 0:
            return None

        results = self.query_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=max_candidates,
            include=["ids", "metadatas", "distances"],
        )

        for cache_id, metadata, dist in zip(results["ids"][0],results["metadatas"][0],results["distances"][0]):
            similarity = 1.0 - dist

            # Monotonic similarity → safe early exit
            if similarity < self.similarity_threshold: # results are returned in descending order so if current fails next all will fail too
                break

            # Expired → delete and continue
            if self._is_expired(metadata):
                self._delete_query(cache_id)
                continue

            # Valid hit
            self._touch(cache_id)
            return cache_id, metadata["payload"], similarity

        return None

    def add_query(self, query: str, embedding: np.ndarray, payload: Dict[str, Any]) -> None:
        """
        - Add a solved query to the cache.
        - Admission policy is enforced by the agent.
        """
        cache_id = str(uuid.uuid4())
        now = time.time()

        metadata = {
            "query": query,
            "created_at": now,
            "payload": payload,
        }

        self.query_collection.add(
            ids=[cache_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
        )

        self._lru[cache_id] = now
        self._evict_if_needed()

    # Tier 2: CHUNK STORE
    def add_chunks(self, chunks: List[str], embeddings: List[np.ndarray], metadatas: List[Dict[str, Any]]) -> None:
        """
        - Store document chunks.
        """

        ids = [str(uuid.uuid4()) for _ in chunks]

        self.chunk_collection.add(
            ids=ids,
            documents=chunks,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadatas,
        )

    def search_chunks(self, query_embedding: np.ndarray, k: int = 8, similarity_threshold: Optional[float] = None, min_chunks: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve anchor chunks (Tier 2).

        Enforces semantic similarity threshold, monotonic early exit, per-document diversity and minimum evidence guard
        """

        if self.chunk_collection.count() == 0:
            return []

        threshold = similarity_threshold or self.similarity_threshold

        results = self.chunk_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        per_doc_counter = defaultdict(int)
        selected: List[Dict[str, Any]] = []

        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist

            if similarity < threshold:
                break

            pmid = meta.get("pmid")
            if pmid is None:
                continue

            if per_doc_counter[pmid] >= self.max_chunks_per_doc:
                continue

            per_doc_counter[pmid] += 1
            selected.append(
                {
                    "text": text,
                    "metadata": meta,
                    "similarity": similarity,
                }
            )

        # Minimum evidence guard (agent-friendly)
        if len(selected) < min_chunks:
            return []

        return selected

    # INTERNAL HELPERS


    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        return (time.time() - metadata["created_at"]) > self.ttl_seconds

    def _touch(self, cache_id: str) -> None:
        self._lru.pop(cache_id, None)
        self._lru[cache_id] = time.time()

    def _evict_if_needed(self) -> None:
        while len(self._lru) > self.max_query_items:
            oldest_id, _ = self._lru.popitem(last=False)
            self._delete_query(oldest_id)

    def _delete_query(self, cache_id: str) -> None:
        try:
            self.query_collection.delete(ids=[cache_id])
        except Exception:
            pass
        self._lru.pop(cache_id, None)
