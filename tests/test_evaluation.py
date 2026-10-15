from docurag.evaluation import (
    keyword_coverage,
    lexical_faithfulness,
    mrr,
    precision_at_k,
    recall_at_k,
)


def test_retrieval_metrics():
    retrieved = ["a", "b", "c"]
    relevant = ["b", "d"]
    assert precision_at_k(retrieved, relevant, 3) == 1 / 3
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert mrr(retrieved, relevant) == 0.5


def test_answer_metrics():
    assert keyword_coverage("The invoice number is A123", ["A123", "invoice"]) == 1.0
    assert lexical_faithfulness("invoice total is 10", ["invoice total is 10 dollars"]) > 0.5
