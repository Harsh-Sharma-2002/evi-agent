from typing import Dict, List, Any, Iterable
from collections import defaultdict


def expand_context_windows(
    anchor_chunks: List[Dict[str, Any]],
    doc_chunks_map: Dict[str, List[Dict[str, Any]]],
    window_size: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Expand context windows around anchor chunks.

    Parameters
    ----------
    anchor_chunks : List[Dict[str, Any]]
        Output of chunk retrieval. Each item must contain:
        - metadata.pmid
        - metadata.chunk_index

    doc_chunks_map : Dict[str, List[Dict[str, Any]]]
        Mapping:
            pmid -> ordered list of all chunks for that document
        Each chunk dict must contain:
        - "chunk_index"
        - "text"

    window_size : int
        Number of neighboring chunks to include on each side.
        window_size=1 → include [idx-1, idx, idx+1]

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Mapping:
            pmid -> ordered list of expanded chunks
    """

    expanded: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)

    for anchor in anchor_chunks:
        meta = anchor.get("metadata", {})
        pmid = meta.get("pmid")
        idx = meta.get("chunk_index")

        if pmid is None or idx is None:
            continue

        if pmid not in doc_chunks_map:
            continue

        all_chunks = doc_chunks_map[pmid]
        total = len(all_chunks)

        start = max(0, idx - window_size)
        end = min(total, idx + window_size + 1)

        for i in range(start, end):
            chunk = all_chunks[i]
            expanded[pmid][chunk["chunk_index"]] = chunk

    # Sort chunks per document by chunk_index
    final_context: Dict[str, List[Dict[str, Any]]] = {}

    for pmid, chunks_by_idx in expanded.items():
        ordered = [
            chunks_by_idx[i]
            for i in sorted(chunks_by_idx.keys())
        ]
        final_context[pmid] = ordered

    return final_context


def flatten_context(
    expanded_context: Dict[str, List[Dict[str, Any]]]
) -> List[str]:
    """
    Flatten expanded context into a list of strings
    suitable for prompt construction.

    Order is preserved within each document.
    Documents are grouped but not interleaved.

    Returns
    -------
    List[str]
        List of chunk texts
    """

    texts: List[str] = []

    for pmid, chunks in expanded_context.items():
        for chunk in chunks:
            texts.append(chunk["text"])

    return texts
