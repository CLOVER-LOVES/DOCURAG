"""PDF loading and chunking utilities for DocuRAG."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf_file(file_path: str | Path, source_name: str | None = None) -> List[Any]:
    """Load a PDF into LangChain documents.

    A missing/empty text extraction raises ``ValueError`` instead of creating
    an unusable, empty vector store.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    has_text = any((getattr(doc, "page_content", "") or "").strip() for doc in documents)
    if documents and has_text:
        source = source_name or path.name
        for index, document in enumerate(documents):
            document.metadata = dict(document.metadata or {})
            document.metadata["source"] = source
            document.metadata.setdefault("page", document.metadata.get("page", index))
        return documents

    # Optional OCR fallback for scanned/image-only PDFs.
    ocr_documents = ocr_pdf(path, source_name or path.name)
    if ocr_documents:
        return ocr_documents

    raise ValueError(
        f"No extractable text found in '{source_name or path.name}'. "
        "The file may be scanned or image-only. Install OCR extras "
        "(pytesseract, pdf2image, and Tesseract) to support it."
    )


def ocr_pdf(path: Path, source_name: str) -> List[Any]:
    """Best-effort OCR for scanned PDFs. Returns [] if OCR is unavailable."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception:
        return []

    try:
        images = convert_from_path(str(path))
    except Exception:
        return []

    from langchain.schema import Document

    docs = []
    for page_number, image in enumerate(images):
        try:
            text = pytesseract.image_to_string(image)
        except Exception:
            text = ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": page_number,
                    },
                )
            )
    return docs


def load_uploaded_pdf(uploaded_file: Any) -> List[Any]:
    """Persist a Streamlit UploadedFile, load it, then remove the temp file."""
    suffix = Path(getattr(uploaded_file, "name", "document.pdf")).suffix or ".pdf"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            data = uploaded_file.read()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            tmp_file.write(data)
            tmp_path = tmp_file.name

        return load_pdf_file(tmp_path, getattr(uploaded_file, "name", None))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def chunk_documents(
    documents: Iterable[Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Any]:
    """Split documents into retrievable chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(list(documents))

    for index, chunk in enumerate(chunks):
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata.setdefault("chunk_id", index)
        chunk.metadata.setdefault("char_length", len(chunk.page_content or ""))

    if not chunks:
        raise ValueError("No text chunks were created from the provided documents.")
    return chunks
