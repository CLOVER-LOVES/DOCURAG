"""Simple query logging and aggregate helpers for lightweight dashboards."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

LOG_FILE = Path("logs") / "queries.jsonl"


def log_query(question: str, response: Dict[str, Any], error: str | None = None) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "question": question,
        "latency_seconds": response.get("latency_seconds") if response else None,
        "source_count": response.get("source_count") if response else None,
        "answer_chars": len(response.get("answer", "") or "") if response else 0,
        "error": error,
    }
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def query_stats(log_path: Path = LOG_FILE) -> Dict[str, Any]:
    if not log_path.exists():
        return {"queries": 0, "avg_latency_s": 0.0, "errors": 0, "avg_sources": 0.0}

    records: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return {"queries": 0, "avg_latency_s": 0.0, "errors": 0, "avg_sources": 0.0}

    latencies = [r.get("latency_seconds") or 0.0 for r in records]
    sources = [r.get("source_count") or 0 for r in records]
    errors = [r for r in records if r.get("error")]

    return {
        "queries": len(records),
        "avg_latency_s": round(sum(latencies) / len(latencies), 3),
        "avg_sources": round(sum(sources) / len(sources), 2),
        "errors": len(errors),
    }
