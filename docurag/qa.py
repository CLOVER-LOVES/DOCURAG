"""Prompting and QA-chain helpers."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

DEFAULT_PROMPT = PromptTemplate(
    template=(
        "You are a document-grounded assistant. Answer using only the supplied context. "
        "If the context does not contain the answer, say you do not know. "
        "Cite the source document in every factual answer.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nGrounded answer with citations:"
    ),
    input_variables=["context", "question"],
)


def build_llm(config: Any) -> ChatOpenAI:
    return ChatOpenAI(
        model_name=config.model_name,
        temperature=config.temperature,
        openai_api_key=config.openai_api_key,
    )


def build_qa_chain(llm: Any, retriever: Any, prompt: PromptTemplate = DEFAULT_PROMPT):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def format_sources(source_documents: List[Any]) -> List[Dict[str, Any]]:
    sources = []
    for doc in source_documents:
        sources.append(
            {
                "source": (doc.metadata or {}).get("source", "Unknown"),
                "page": (doc.metadata or {}).get("page", "N/A"),
                "chunk_id": (doc.metadata or {}).get("chunk_id", "N/A"),
                "preview": (doc.page_content or "")[:320],
            }
        )
    return sources


def ask_question(chain: Any, question: str) -> Dict[str, Any]:
    started = time.perf_counter()
    response = chain.invoke({"query": question})
    elapsed = time.perf_counter() - started

    return {
        "answer": response.get("result", ""),
        "sources": format_sources(response.get("source_documents", [])),
        "latency_seconds": round(elapsed, 3),
        "source_count": len(response.get("source_documents", [])),
    }
