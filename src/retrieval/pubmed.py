from ..agent.state import AgentState
import requests
import numpy as np
from typing import List, Dict, Any
from retrieval.cache import VectorCache
import xml.etree.ElementTree as ET
from utils.llm import embed


def _fetch_pubmed_docs(query: str, retmax: int = 5, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Fetch PubMed documents.

    Returns:
    {
      "pmid": str,
      "year": int | None,
      "abstract": str | None,
      "conclusion": str | None
    }
    """

    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "retstart": offset,
    }

    pmids = requests.get(search_url, params=search_params, timeout=10)\
                     .json()["esearchresult"]["idlist"]

    if not pmids:
        return []

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    root = ET.fromstring(
        requests.get(fetch_url, params=fetch_params, timeout=10).text
    )

    docs: List[Dict[str, Any]] = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        year_text = article.findtext(".//PubDate/Year")

        abstract = " ".join(
            t.text for t in article.findall(".//Abstract/AbstractText")
            if t.text
        )

        # Conclusion: explicit section or fallback to last AbstractText
        conclusion_nodes = [
            t.text for t in article.findall(".//Abstract/AbstractText[@Label='CONCLUSION']")
            if t.text
        ]

        conclusion = (
            conclusion_nodes[0]
            if conclusion_nodes
            else None
        )

        if not pmid or not abstract:
            continue

        docs.append(
            {
                "pmid": pmid,
                "year": int(year_text) if year_text and year_text.isdigit() else None,
                "abstract": abstract.strip(),
                "conclusion": conclusion.strip() if conclusion else None,
            }
        )

    return docs



def pubmed_fetch_node(state: AgentState, cache: VectorCache) -> AgentState:
    """
    Tier 3:
    - Fetch PubMed documents
    - Create two chunks per paper:
        0 → abstract
        1 → conclusion (if available)
    """
    offset = state["iteration"] * 5
    documents = _fetch_pubmed_docs(query=state["query"],retmax=5,offset=offset)

    if not documents:
        state["api_calls"] += 1
        return state
    
    chunk_texts: List[str] = []
    chunk_embed = List[np.ndarray]
    chunk_meta = List[Dict[str,Any]] = []

    for doc in documents:
        pmid = doc["pmid"]
        year = doc["year"]
        
        # Abstract chunk
        abs_txt = doc["abstract"]
        emb = np.asarray(embed(abs_txt))

        chunk_texts.append(abs_txt)
        chunk_embed.append(emb)
        chunk_meta.append({"pmid":pmid,
                          "year":year,
                          "chunk_index": 0,
                          "section": "abstract"
                        })
        state["doc_chunks_map"].setdefault(pmid,[]).append({
            "text" : abs_txt,
            "chunk_index": 0,
            "metadata": {"pmid":pmid, "year": year}
        })

        if doc["conclusion"]:
            con_txt = doc["conclusion"]
            emb = np.asarray(embed(con_txt))

            chunk_texts.append(con_txt)
            chunk_embed.append(emb)
            chunk_meta.append({
                "pmid": pmid,
                "year": year,
                "chunk_index": 1,
                "section": "conclusion"
            })

            state["doc_chunks_map"][pmid].append({
                "text": con_txt,
                "chunk_index": 1,
                "metadata": {"pmid": pmid, "year": year},
            })

    cache.add_chunks(chunk_texts, chunk_embed, chunk_meta)

    state["documents"].extend(documents)
    state["api_calls"] += 1

    return state

        

