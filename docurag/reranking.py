"""Optional cross-encoder reranking for stronger retrieval precision."""
from __future__ import annotations

import time
from typing import Any, List, Tuple


def reranking_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except Exception:
        return False


class RerankingRetriever:
    """Wrap a retriever and rerank its candidate documents with a cross-encoder."""

    def __init__(self, base_retriever: Any, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 4):
        self.base_retriever = base_retriever
        self.model_name = model_name
        self.top_k = top_k
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    "Cross-encoder reranking requires 'sentence-transformers'. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
        return self._model

    def invoke(self, query: str) -> List[Any]:
        docs = self.base_retriever.invoke(query)
        if not docs:
            return docs

        # Retrieve a wider candidate set when the base retriever uses top-k.
        model = self._load_model()
        pairs = [(query, getattr(doc, "page_content", "") or "") for doc in docs]
        scores = model.predict(pairs)
        ranked: List[Tuple[Any, float]] = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]

    def get_relevant_documents(self, query: str) -> List[Any]:
        return self.invoke(query)


def maybe_wrap_reranker(retriever: Any, enabled: bool, model_name: str, top_k: int):
    if not enabled:
        return retriever
    if not reranking_available():
        raise RuntimeError(
            "Reranking requested, but sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )
    return RerankingRetriever(retriever, model_name=model_name, top_k=top_k)
