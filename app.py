import streamlit as st
import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
try:
    from langchain_community.llms import Ollama
except ImportError:
    from langchain_ollama import Ollama  # langchain-ollama>=0.1 fallback
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import hashlib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap');

/* Root variables */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #18181f;
    --border: #2a2a35;
    --accent: #7c6aff;
    --accent2: #ff6a88;
    --text: #e8e8f0;
    --muted: #6b6b80;
    --success: #4ade80;
    --radius: 12px;
}

/* Global reset */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    color: var(--text);
}

.stApp {
    background: var(--bg);
}

/* Hide streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    background: var(--surface2);
    transition: border-color 0.3s;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #9b8bff) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(124,106,255,0.35) !important;
}

/* Text input */
.stTextInput > div > div > input,
[data-testid="stChatInput"] > div > div > textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
}

[data-testid="stChatInput"] > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,106,255,0.2) !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 0.75rem !important;
    padding: 0.75rem 1rem !important;
}

/* User messages */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid var(--accent2) !important;
}

/* Assistant messages */
.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid var(--accent) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Logo header */
.logo-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.5rem;
}

.logo-glyph {
    font-size: 2rem;
    line-height: 1;
}

.logo-text {
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.logo-sub {
    font-size: 0.7rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.3);
    color: #4ade80;
    padding: 3px 10px;
    border-radius: 50px;
}

.status-pill.loading {
    background: rgba(124,106,255,0.1);
    border-color: rgba(124,106,255,0.3);
    color: var(--accent);
}

.status-pill.error {
    background: rgba(255,106,136,0.1);
    border-color: rgba(255,106,136,0.3);
    color: var(--accent2);
}

/* Doc card */
.doc-card {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 0.82rem;
    font-family: 'DM Mono', monospace;
    color: var(--muted);
}

.doc-card .doc-icon { color: var(--accent); font-size: 1rem; }
.doc-card .doc-name { flex: 1; color: var(--text); font-size: 0.8rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.doc-card .doc-pages { color: var(--muted); font-size: 0.7rem; }

/* Section label */
.section-label {
    font-size: 0.65rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
    margin-top: 16px;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1.5rem;
    color: var(--muted);
}
.empty-state .big-icon { font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-state h3 { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text); opacity: 0.6; }
.empty-state p { font-size: 0.85rem; font-family: 'DM Mono', monospace; line-height: 1.6; }

/* Source expander */
.source-badge {
    display: inline-block;
    background: rgba(124,106,255,0.15);
    border: 1px solid rgba(124,106,255,0.3);
    color: var(--accent);
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px;
}

/* Stats row */
.stats-row {
    display: flex;
    gap: 12px;
    margin-bottom: 1.2rem;
}
.stat-box {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    text-align: center;
}
.stat-val {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-lbl {
    font-size: 0.65rem;
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "chat_history": [],
        "vectorstore": None,
        "chain": None,
        "memory": None,
        "doc_metadata": [],
        "processed_hashes": set(),
        "total_chunks": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def load_and_chunk(pdf_path: str, chunk_size: int, chunk_overlap: int):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(pages), len(pages)


def build_chain(vectorstore, llm_choice: str, api_key: str, temperature: float, k: int):
    if llm_choice == "OpenAI GPT-4o":
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            openai_api_key=api_key,
        )
    elif llm_choice == "OpenAI GPT-3.5":
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=temperature,
            openai_api_key=api_key,
        )
    else:  # Ollama local
        llm = Ollama(model="llama3", temperature=temperature)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain, memory


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='logo-header'>
        <span class='logo-glyph'>🧠</span>
        <div>
            <div class='logo-text'>DocMind</div>
            <div class='logo-sub'>RAG Chatbot</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>📄 Upload Documents</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("<div class='section-label'>🤖 LLM Provider</div>", unsafe_allow_html=True)
    llm_choice = st.selectbox(
        "LLM",
        ["OpenAI GPT-4o", "OpenAI GPT-3.5", "Ollama (Local llama3)"],
        label_visibility="collapsed",
    )

    api_key = ""
    if "OpenAI" in llm_choice:
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if not api_key:
            st.markdown("<div class='status-pill error'>⚠ API key required</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>⚙ Retrieval Settings</div>", unsafe_allow_html=True)

    with st.expander("Advanced", expanded=False):
        chunk_size = st.slider("Chunk size (tokens)", 256, 2048, 800, 64)
        chunk_overlap = st.slider("Chunk overlap", 0, 512, 150, 16)
        top_k = st.slider("Top-K chunks retrieved", 1, 10, 4)
        temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.05)

    if uploaded_files:
        process_btn = st.button("⚡ Process Documents", use_container_width=True)
    else:
        st.button("⚡ Process Documents", use_container_width=True, disabled=True)
        process_btn = False

    # ── Process docs ─────────────────────────────────────────────────────────
    if process_btn and uploaded_files:
        embeddings = get_embeddings()
        new_docs = []
        progress = st.progress(0, text="Reading PDFs…")

        for i, uf in enumerate(uploaded_files):
            raw = uf.read()
            fh = file_hash(raw)

            if fh in st.session_state.processed_hashes:
                st.toast(f"↩ Already indexed: {uf.name}", icon="ℹ️")
                continue

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

            chunks, n_pages = load_and_chunk(tmp_path, chunk_size, chunk_overlap)
            os.unlink(tmp_path)

            new_docs.extend(chunks)
            st.session_state.processed_hashes.add(fh)
            st.session_state.doc_metadata.append({
                "name": uf.name,
                "pages": n_pages,
                "chunks": len(chunks),
            })
            progress.progress((i + 1) / len(uploaded_files), text=f"Chunked {uf.name}")

        if new_docs:
            st.session_state.total_chunks += len(new_docs)
            progress.progress(0.9, text="Building vector index…")

            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = FAISS.from_documents(new_docs, embeddings)
            else:
                st.session_state.vectorstore.add_documents(new_docs)

            progress.progress(1.0, text="Done!")
            st.session_state.memory = None  # reset memory on new docs

            if "OpenAI" in llm_choice and not api_key:
                st.error("Please enter your OpenAI API key.")
            else:
                try:
                    chain, memory = build_chain(
                        st.session_state.vectorstore, llm_choice, api_key, temperature, top_k
                    )
                    st.session_state.chain = chain
                    st.session_state.memory = memory
                    st.success(f"✓ Indexed {len(new_docs)} new chunks!")
                except Exception as e:
                    st.error(f"Chain error: {e}")

    # ── Doc list ─────────────────────────────────────────────────────────────
    if st.session_state.doc_metadata:
        st.markdown("<div class='section-label'>📚 Indexed Documents</div>", unsafe_allow_html=True)
        for dm in st.session_state.doc_metadata:
            st.markdown(f"""
            <div class='doc-card'>
                <span class='doc-icon'>📄</span>
                <span class='doc-name' title='{dm["name"]}'>{dm["name"]}</span>
                <span class='doc-pages'>{dm["pages"]}p · {dm["chunks"]}c</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear All", use_container_width=True):
        for k in ["chat_history", "vectorstore", "chain", "memory", "doc_metadata", "total_chunks"]:
            st.session_state[k] = [] if k in ["chat_history", "doc_metadata"] else None
        st.session_state.processed_hashes = set()
        st.session_state.total_chunks = 0
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("## Ask Your Documents")
with col_status:
    if st.session_state.chain:
        st.markdown("<div class='status-pill'>● Ready</div>", unsafe_allow_html=True)
    elif st.session_state.vectorstore:
        st.markdown("<div class='status-pill loading'>◌ Building chain…</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-pill error'>○ No docs loaded</div>", unsafe_allow_html=True)

# Stats row
if st.session_state.doc_metadata:
    total_pages = sum(d["pages"] for d in st.session_state.doc_metadata)
    st.markdown(f"""
    <div class='stats-row'>
        <div class='stat-box'><div class='stat-val'>{len(st.session_state.doc_metadata)}</div><div class='stat-lbl'>Docs</div></div>
        <div class='stat-box'><div class='stat-val'>{total_pages}</div><div class='stat-lbl'>Pages</div></div>
        <div class='stat-box'><div class='stat-val'>{st.session_state.total_chunks}</div><div class='stat-lbl'>Chunks</div></div>
        <div class='stat-box'><div class='stat-val'>{len(st.session_state.chat_history)}</div><div class='stat-lbl'>Turns</div></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Chat display ──────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
        <div class='empty-state'>
            <div class='big-icon'>💬</div>
            <h3>Start a Conversation</h3>
            <p>Upload PDFs in the sidebar, process them,<br>then ask anything about your documents.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])
                if role == "assistant" and msg.get("sources"):
                    with st.expander("📎 Source chunks", expanded=False):
                        for src in msg["sources"]:
                            pg = src.metadata.get("page", "?")
                            src_file = src.metadata.get("source", "doc")
                            st.markdown(
                                f"<span class='source-badge'>📄 {os.path.basename(src_file)} · p.{pg}</span>",
                                unsafe_allow_html=True,
                            )
                            st.caption(src.page_content[:350] + "…")

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask something about your documents…")

if user_input:
    if not st.session_state.chain:
        st.warning("⚠ Please upload and process PDF documents first.")
    else:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = st.session_state.chain({"question": user_input})
                    answer = result["answer"]
                    sources = result.get("source_documents", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("📎 Source chunks", expanded=False):
                            for src in sources:
                                pg = src.metadata.get("page", "?")
                                src_file = src.metadata.get("source", "doc")
                                st.markdown(
                                    f"<span class='source-badge'>📄 {os.path.basename(src_file)} · p.{pg}</span>",
                                    unsafe_allow_html=True,
                                )
                                st.caption(src.page_content[:350] + "…")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": err, "sources": []
                    })
