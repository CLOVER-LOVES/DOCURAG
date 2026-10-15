"""Vector, keyword, and hybrid retrieval helpers."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


class TimedRetriever:
    """Small wrapper that records retrieval latency for UI/monitoring."""

    def __init__(self, retriever: Any, name: str = "retriever"):
        self.retriever = retriever
        self.name = name

    def invoke(self, query: str) -> Tuple[List[Any], float]:
        started = time.perf_counter()
        docs = self.retriever.invoke(query)
        elapsed = time.perf_counter() - started
        return docs, elapsed

    def get_relevant_documents(self, query: str) -> List[Any]:
        return self.retriever.get_relevant_documents(query)


def make_bm25_retriever(chunks: List[Any], k: int = 4) -> BM25Retriever:
    """Create a sparse keyword retriever over the same chunks."""
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


def make_hybrid_retriever(
    vector_retriever: Any,
    chunks: List[Any],
    k: int = 4,
    bm25_weight: float = 0.35,
):
    """Combine dense vector search with BM25 keyword search.

    This is a strong, practical upgrade over pure embedding retrieval for
    names, IDs, acronyms, and exact phrases.
    """
    if not 0 < bm25_weight < 1:
        raise ValueError("bm25_weight must be between 0 and 1")

    bm25 = make_bm25_retriever(chunks, k=k)
    vector_retriever.search_kwargs = {**(vector_retriever.search_kwargs or {}), "k": k}
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[1 - bm25_weight, bm25_weight],
    )


def make_retriever(
    vector_store: Any,
    chunks: List[Any],
    config: Any,
) -> Any:
    """Create the configured retriever."""
    base = vector_store.as_retriever(search_kwargs={"k": config.top_k})

    if getattr(config, "retrieval_mode", "dense") == "hybrid":
        return make_hybrid_retriever(base, chunks, k=config.top_k, bm25_weight=config.bm25_weight)

    return base


def persist_vector_store(vector_store: Any, persist_directory: str | Path) -> None:
    """Persist Chroma using whichever API the installed version supports."""
    persist_directory = str(persist_directory)
    if hasattr(vector_store, "persist"):
        vector_store.persist()
    # Newer Chroma versions persist automatically by path.
    _ = persist_directory
