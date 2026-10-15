from langchain.schema import Document

from docurag.ingestion import chunk_documents


def test_chunk_documents_adds_metadata():
    docs = [Document(page_content="a " * 300, metadata={"source": "test.pdf", "page": 1})]
    chunks = chunk_documents(docs, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all("chunk_id" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["source"] == "test.pdf" for chunk in chunks)


def test_chunk_validation():
    try:
        chunk_documents([Document(page_content="short", metadata={})], chunk_size=10, chunk_overlap=10)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid overlap should raise ValueError")
