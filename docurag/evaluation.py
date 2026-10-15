"""Lightweight RAG evaluation helpers and benchmark runner."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall((text or "").lower()))


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_k = [str(x).lower() for x in retrieved[:k]]
    relevant_set = {str(x).lower() for x in relevant}
    if not retrieved_k:
        return 0.0
    hits = sum(1 for item in retrieved_k if item in relevant_set)
    return hits / min(k, len(retrieved_k))


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 1.0
    retrieved_set = {str(x).lower() for x in retrieved[:k]}
    relevant_set = {str(x).lower() for x in relevant}
    hits = len(retrieved_set & relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    relevant_set = {str(x).lower() for x in relevant}
    for rank, item in enumerate(retrieved, start=1):
        if str(item).lower() in relevant_set:
            return 1.0 / rank
    return 0.0


def keyword_coverage(answer: str, expected_keywords: Iterable[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_tokens = tokenize(answer)
    total = 0
    for keyword in expected_keywords:
        keyword_tokens = tokenize(keyword)
        if keyword_tokens and keyword_tokens <= answer_tokens:
            total += 1
        elif not keyword_tokens:
            total += 1
    return total / len(list(expected_keywords))


def lexical_faithfulness(answer: str, contexts: Iterable[str]) -> float:
    """A cheap proxy metric based on answer/context token overlap."""
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 1.0
    context_tokens = set()
    for context in contexts:
        context_tokens |= tokenize(context)
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def evaluate_retrieval(
    retriever: Any,
    examples: List[Dict[str, Any]],
    k: int = 4,
) -> Dict[str, float]:
    precisions = []
    recalls = []
    mrrs = []
    latencies = []

    for example in examples:
        start = time.perf_counter()
        docs = retriever.invoke(example["question"])
        latencies.append(time.perf_counter() - start)

        expected_sources = example.get("relevant_chunks") or example.get("relevant_sources", [])
        retrieved = []
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            retrieved.append(metadata.get("chunk_id", metadata.get("source", "")))

        precisions.append(precision_at_k(retrieved, expected_sources, k))
        recalls.append(recall_at_k(retrieved, expected_sources, k))
        mrrs.append(mrr(retrieved, expected_sources))

    n = max(len(examples), 1)
    return {
        "precision@k": round(sum(precisions) / n, 4),
        "recall@k": round(sum(recalls) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "avg_retrieval_latency_s": round(sum(latencies) / n, 4),
    }


def evaluate_pipeline(
    qa_chain: Any,
    examples: List[Dict[str, Any]],
    k: int = 4,
) -> Dict[str, Any]:
    retrieval_metrics = {
        "precision@k": 0.0,
        "recall@k": 0.0,
        "mrr": 0.0,
    }
    keyword_scores = []
    faithfulness_scores = []
    latencies = []

    for example in examples:
        from docurag.qa import ask_question

        response = ask_question(qa_chain, example["question"])
        answer = response["answer"]
        latency = response["latency_seconds"]
        sources = response["sources"]
        contexts = [source["preview"] for source in sources]

        score = keyword_coverage(answer, example.get("expected_keywords", []))
        keyword_scores.append(score)
        faithfulness_scores.append(lexical_faithfulness(answer, contexts))
        latencies.append(latency)

    n = max(len(examples), 1)
    return {
        "answer_keyword_coverage": round(sum(keyword_scores) / n, 4),
        "lexical_faithfulness": round(sum(faithfulness_scores) / n, 4),
        "avg_end_to_end_latency_s": round(sum(latencies) / n, 4),
    }


def load_eval_dataset(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must be a JSON list.")
    return data
