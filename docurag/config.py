"""Runtime configuration shared by the app and evaluation scripts."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict


@dataclass
class RAGConfig:
    """Configuration for document ingestion and retrieval."""

    openai_api_key: str
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    retrieval_mode: str = "hybrid"  # "dense" or "hybrid"
    bm25_weight: float = 0.35
    persist_directory: Path = Path("chroma_db")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["persist_directory"] = str(self.persist_directory)
        return data

    @classmethod
    def from_streamlit(cls, api_key: str, sidebar: Dict[str, Any], base_dir: Path) -> "RAGConfig":
        return cls(
            openai_api_key=api_key,
            model_name=sidebar["model_name"],
            temperature=sidebar["temperature"],
            chunk_size=sidebar["chunk_size"],
            chunk_overlap=sidebar["chunk_overlap"],
            top_k=sidebar["k_docs"],
            retrieval_mode=sidebar.get("retrieval_mode", "hybrid"),
            bm25_weight=sidebar.get("bm25_weight", 0.35),
            persist_directory=base_dir / "chroma_db",
        )
