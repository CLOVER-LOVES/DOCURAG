import streamlit as st
import os
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"
load_dotenv(BASE_DIR / ".env")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from docurag.config import RAGConfig
from docurag.ingestion import load_uploaded_pdf, chunk_documents
from docurag.retrieval import make_retriever, persist_vector_store
from docurag.reranking import maybe_wrap_reranker, reranking_available
from docurag.qa import build_llm, build_qa_chain, ask_question
from docurag.monitoring import log_query, query_stats

st.set_page_config(
    page_title="DocuChat - RAG Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS (Framer Motion-inspired micro-interactions)
st.markdown("""
<style>
    :root {
        --spring: cubic-bezier(0.16, 1, 0.3, 1);
        --bg-a: #070b1a;
        --bg-b: #111827;
        --glass: rgba(255, 255, 255, 0.06);
        --glass-border: rgba(255, 255, 255, 0.12);
        --accent: #60a5fa;
        --accent-2: #a78bfa;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes floatBlobs {
        from { transform: translate3d(0, 0, 0) scale(1); }
        to { transform: translate3d(2%, 3%, 0) scale(1.05); }
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 0 rgba(96, 165, 250, 0.35); }
        50% { box-shadow: 0 0 30px rgba(167, 139, 250, 0.45); }
    }

    html, body, [class*="css"] {
        font-family: "Inter", "SF Pro Display", system-ui, -apple-system, sans-serif;
    }

    .stApp {
        background: linear-gradient(120deg, var(--bg-a), #1e1b4b, #312e81, var(--bg-b));
        background-size: 300% 300%;
        animation: gradientShift 20s ease infinite;
        color: #e5e7eb;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(circle at 15% 20%, rgba(56, 189, 248, 0.16), transparent 36%),
            radial-gradient(circle at 85% 15%, rgba(168, 85, 247, 0.18), transparent 40%),
            radial-gradient(circle at 50% 85%, rgba(244, 114, 182, 0.14), transparent 42%);
        animation: floatBlobs 14s ease-in-out infinite alternate;
    }

    .block-container {
        position: relative;
        z-index: 1;
        padding-top: 2rem;
        animation: fadeInUp 0.75s var(--spring) both;
    }

    .main-header {
        font-size: clamp(2.1rem, 5vw, 3.6rem);
        font-weight: 800;
        text-align: center;
        letter-spacing: -0.04em;
        margin-bottom: 0.25rem;
        background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6, #38bdf8);
        background-size: 220% 220%;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: shimmer 8s linear infinite, fadeInDown 0.85s var(--spring) both;
    }

    .sub-header {
        text-align: center;
        color: rgba(226, 232, 240, 0.74);
        margin-bottom: 2rem;
        animation: fadeInUp 0.95s var(--spring) both;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.58);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--glass-border);
        animation: fadeInDown 0.6s var(--spring) both;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        background: var(--glass);
        backdrop-filter: blur(18px);
        border: 1px solid var(--glass-border);
        border-radius: 22px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
        transition: transform 0.3s var(--spring), box-shadow 0.3s var(--spring), border-color 0.3s ease;
        animation: fadeInUp 0.7s var(--spring) both;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:hover {
        transform: translateY(-4px);
        border-color: rgba(96, 165, 250, 0.35);
        box-shadow: 0 28px 80px rgba(0, 0, 0, 0.38);
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: #ffffff;
        border: none;
        border-radius: 14px;
        font-weight: 600;
        padding: 0.72rem 1rem;
        box-shadow: 0 14px 34px rgba(37, 99, 235, 0.32);
        transition: transform 0.2s var(--spring), box-shadow 0.2s var(--spring), filter 0.2s ease;
        animation: pulseGlow 4s ease-in-out infinite;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 20px 46px rgba(124, 58, 237, 0.45);
        filter: brightness(1.08);
    }
    .stButton > button:active {
        transform: translateY(1px) scale(0.985);
    }

    .stTextInput input,
    .stChatInput textarea,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.55) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
        border-radius: 12px !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s var(--spring);
    }
    .stTextInput input:focus,
    .stChatInput textarea:focus,
    .stTextArea textarea:focus {
        border-color: rgba(96, 165, 250, 0.7) !important;
        box-shadow: 0 0 0 4px rgba(96, 165, 250, 0.18) !important;
        transform: translateY(-1px);
    }

    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 0.75rem 1rem;
        margin: 0.55rem 0;
        animation: fadeInUp 0.45s var(--spring) both;
        transition: transform 0.2s var(--spring), background 0.2s ease;
    }
    [data-testid="stChatMessage"]:hover {
        transform: translateX(4px);
        background: rgba(255, 255, 255, 0.075);
    }

    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        overflow: hidden;
        animation: fadeInUp 0.5s var(--spring) both;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px dashed rgba(96, 165, 250, 0.45);
        border-radius: 18px;
        padding: 0.75rem;
        transition: border-color 0.25s ease, transform 0.25s var(--spring);
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(167, 139, 250, 0.7);
        transform: scale(1.005);
    }

    .metric-chip {
        display: inline-flex;
        gap: 0.5rem;
        align-items: center;
        padding: 0.35rem 0.75rem;
        margin: 0.15rem;
        background: rgba(96, 165, 250, 0.1);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 999px;
        font-size: 0.82rem;
        color: #bfdbfe;
        animation: fadeInUp 0.4s var(--spring) both;
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📚 DocuChat</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-leaning document Q&A with hybrid retrieval and grounded citations</div>', unsafe_allow_html=True)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    model_name = st.selectbox(
        "Select Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)

    retrieval_mode = st.radio(
        "Retrieval Strategy",
        ["hybrid", "dense"],
        horizontal=True,
        help="Hybrid combines dense vectors with BM25 keyword search.",
    )
    bm25_weight = st.slider(
        "BM25 Weight",
        0.05,
        0.8,
        0.35,
        0.05,
        disabled=retrieval_mode != "hybrid",
    )

    use_reranking = st.checkbox(
        "Cross-encoder reranking",
        value=False,
        disabled=not reranking_available(),
        help="Requires optional sentence-transformers installation.",
    )
    if not reranking_available():
        st.caption("Install sentence-transformers to enable reranking.")

    with st.expander("📄 Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
        k_docs = st.slider("Documents to Retrieve", 1, 10, 4, 1)

    st.divider()

    if st.button("🗑️ Clear Database"):
        if DB_DIR.exists():
            shutil.rmtree(DB_DIR)
        st.session_state.vector_store = None
        st.session_state.qa_chain = None
        st.session_state.processed_files = []
        st.session_state.chunks = []
        st.session_state.chat_history = []
        st.session_state.last_metrics = None
        st.success("Database cleared!")
        st.rerun()

    stats = query_stats()
    with st.expander("📊 Query Dashboard", expanded=False):
        st.write(f"Queries: {stats['queries']}")
        st.write(f"Average latency: {stats['avg_latency_s']}s")
        st.write(f"Average sources: {stats['avg_sources']}")
        st.write(f"Errors: {stats['errors']}")

    if st.session_state.processed_files:
        st.success(f"📁 {len(st.session_state.processed_files)} file(s) indexed")
        with st.expander("View Files"):
            for file_name in st.session_state.processed_files:
                st.write(f"✓ {file_name}")

    if st.session_state.last_metrics:
        st.divider()
        st.caption("Last query metrics")
        metrics = st.session_state.last_metrics
        st.markdown(
            f'<span class="metric-chip">⏱ {metrics["latency_seconds"]}s</span>'
            f'<span class="metric-chip">📑 {metrics["source_count"]} sources</span>',
            unsafe_allow_html=True,
        )

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more text-based PDF files.",
    )

    if uploaded_files and api_key:
        if st.button("🚀 Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    all_documents = []
                    for uploaded_file in uploaded_files:
                        docs = load_uploaded_pdf(uploaded_file)
                        all_documents.extend(docs)
                        if uploaded_file.name not in st.session_state.processed_files:
                            st.session_state.processed_files.append(uploaded_file.name)

                    chunks = chunk_documents(
                        all_documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    st.session_state.chunks = chunks

                    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                    DB_DIR.mkdir(parents=True, exist_ok=True)

                    if st.session_state.vector_store is None:
                        st.session_state.vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=str(DB_DIR),
                        )
                    else:
                        st.session_state.vector_store.add_documents(chunks)
                        persist_vector_store(st.session_state.vector_store, DB_DIR)

                    config = RAGConfig.from_streamlit(
                        api_key,
                        {
                            "model_name": model_name,
                            "temperature": temperature,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "k_docs": k_docs,
                            "retrieval_mode": retrieval_mode,
                            "bm25_weight": bm25_weight,
                        },
                        BASE_DIR,
                    )

                    retriever = make_retriever(
                        st.session_state.vector_store,
                        chunks,
                        config,
                    )
                    retriever = maybe_wrap_reranker(
                        retriever,
                        use_reranking,
                        "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        k_docs,
                    )
                    llm = build_llm(config)
                    st.session_state.qa_chain = build_qa_chain(llm, retriever)

                    st.success(
                        f"✅ Processed {len(all_documents)} page(s) into "
                        f"{len(chunks)} chunk(s) using {retrieval_mode} retrieval."
                    )

                except Exception as e:
                    error_text = str(e)
                    lower = error_text.lower()
                    if "credit_balance_exhausted" in lower or "insufficient_quota" in lower:
                        st.error("OpenAI API quota exhausted. Add billing/credits, then retry.")
                    elif "invalid_api_key" in lower or "incorrect api key" in lower:
                        st.error("Invalid OpenAI API key. Check the sidebar value.")
                    elif "no extractable text" in lower:
                        st.error(error_text)
                    else:
                        st.error(f"Error processing documents: {error_text}")
    elif uploaded_files and not api_key:
        st.warning("⚠️ Enter an OpenAI API key in the sidebar")

with col2:
    st.subheader("💬 Ask Questions")

    chat_container = st.container()
    with chat_container:
        for question, answer in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)

    if st.session_state.qa_chain:
        question = st.chat_input("Ask a question about your documents...")

        if question:
            with st.spinner("Thinking..."):
                try:
                    response = ask_question(st.session_state.qa_chain, question)
                    answer = response["answer"]
                    sources = response["sources"]
                    st.session_state.chat_history.append((question, answer))
                    st.session_state.last_metrics = response
                    log_query(question, response)

                    with st.chat_message("user"):
                        st.write(question)
                    with st.chat_message("assistant"):
                        st.write(answer)

                        if sources:
                            with st.expander("📑 View Sources"):
                                for index, source in enumerate(sources, start=1):
                                    st.markdown(
                                        f"**Source {index}:** {source['source']}"
                                    )
                                    st.markdown(f"*Page {source['page']} · Chunk {source['chunk_id']}*")
                                    st.text(source["preview"] + ("..." if len(source["preview"]) >= 320 else ""))
                                    st.divider()

                except Exception as e:
                    log_query(question, {}, error=str(e))
                    st.error(f"Error: {str(e)}")
    else:
        st.info("👆 Upload and process documents first to start asking questions")

st.divider()
st.markdown("""
<div style='text-align: center; color: rgba(226,232,240,0.55); padding: 1rem;'>
    <p>Built with LangChain, ChromaDB, OpenAI, and Streamlit</p>
    <p>💡 Tip: Ask specific, source-verifiable questions for best results</p>
</div>
""", unsafe_allow_html=True)
