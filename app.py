"""
app.py — Streamlit front end for the Textbook Doubt Solver.

Run with:   streamlit run app.py

This file only handles the screen (upload box, chat, buttons).
All the AI work happens in rag_pipeline.py.
"""

import html

import streamlit as st
from dotenv import load_dotenv

import rag_pipeline as rag
from utils.styles import CUSTOM_CSS

load_dotenv()  # reads GEMINI_API_KEY (and optional GEMINI_MODEL) from .env

st.set_page_config(
    page_title="Textbook Doubt Solver",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state: Streamlit re-runs this script on every click, so anything we
# want to remember (chat history, FAISS index...) is stored here.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "messages": [],        # chat history (list of dicts)
    "doc_id": None,        # identifies the currently loaded PDF
    "doc_name": None,
    "pages": None,
    "chunks": None,
    "index": None,
    "doc_error": None,
    "pending_simplify": False,
    "sidebar_api_key": "",
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load the Sentence Transformer only once per server (it is slow to load)."""
    return rag.load_embedding_model()


def current_api_key():
    return rag.get_api_key() or (st.session_state.sidebar_api_key or "").strip() or None


def reset_document():
    for key in ("doc_id", "doc_name", "pages", "chunks", "index", "doc_error"):
        st.session_state[key] = None
    st.session_state.messages = []


def clear_chat():
    st.session_state.messages = []
    st.session_state.pending_simplify = False


def last_answer_message():
    """Most recent AI answer that can be simplified (not a 'not found' reply)."""
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and msg.get("found") and not msg.get("error"):
            return msg
    return None


# ---------------------------------------------------------------------------
# Sidebar: project info, instructions, API key status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-title">📘 Textbook Doubt Solver</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <table class="info-table">
          <tr><td>Problem No.</td><td>33</td></tr>
          <tr><td>Roll No.</td><td>33</td></tr>
          <tr><td>Domain</td><td>Generative AI / LLMs</td></tr>
          <tr><td>Architecture</td><td>RAG</td></tr>
          <tr><td>LLM</td><td>Google Gemini</td></tr>
          <tr><td>Embeddings</td><td>Sentence Transformers<br>(all-MiniLM-L6-v2)</td></tr>
          <tr><td>Vector DB</td><td>FAISS</td></tr>
          <tr><td>Frontend</td><td>Streamlit</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**How to use**")
    st.markdown(
        "1. Upload a textbook chapter (PDF).\n"
        "2. Wait for **Ready to answer** to appear.\n"
        "3. Type your doubt in the box at the bottom.\n"
        "4. Read the answer, quoted source line and page number.\n"
        "5. Click **Explain Simpler** for an easier version."
    )
    st.divider()
    st.markdown("**Gemini API key**")
    if rag.get_api_key():
        st.success("Loaded from .env file ✅")
    else:
        st.warning("No key in .env. Paste one below (kept only for this session).")
        st.text_input(
            "Gemini API key", type="password", key="sidebar_api_key",
            placeholder="Paste key from Google AI Studio",
        )
    st.caption(f"Model: `{rag.get_model_name()}`")
    st.divider()
    st.caption("Answers come only from the uploaded PDF. Always cross-check important facts with your textbook.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>📘 Textbook Doubt Solver</h1>
      <p>Your AI-powered study companion.</p>
      <span class="badge">RAG</span><span class="badge">Gemini</span>
      <span class="badge">FAISS</span><span class="badge">Sentence Transformers</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# PDF upload + processing
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="card"><h3>1 · Upload your textbook chapter</h3>'
    '<div class="muted">Text-based PDF files work best. Scanned images cannot be read.</div></div>',
    unsafe_allow_html=True,
)
uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

if uploaded is None:
    if st.session_state.doc_id is not None:
        reset_document()  # file was removed from the uploader
else:
    doc_id = f"{uploaded.name}-{uploaded.size}"
    if doc_id != st.session_state.doc_id:
        # A new PDF: forget the old one and process this one.
        reset_document()
        st.session_state.doc_id = doc_id
        st.session_state.doc_name = uploaded.name
        with st.status(f"Processing **{uploaded.name}** …", expanded=True) as status:
            try:
                st.write("🧠 Loading embedding model (first run downloads ~90 MB)…")
                model = get_embedding_model()
                st.write("📄 Extracting text page by page…")
                pages = rag.extract_text_from_pdf(uploaded)
                st.write(f"✂️ Splitting {len(pages)} pages into chunks…")
                chunks = rag.split_text_into_chunks(pages)
                st.write(f"🔢 Creating embeddings for {len(chunks)} chunks…")
                embeddings = rag.create_embeddings([c["text"] for c in chunks], model)
                st.write("🗂️ Building FAISS index…")
                index = rag.build_faiss_index(embeddings)

                st.session_state.pages = pages
                st.session_state.chunks = chunks
                st.session_state.index = index
                status.update(label="Textbook processed successfully", state="complete", expanded=False)
            except rag.PDFProcessingError as exc:
                st.session_state.doc_error = str(exc)
                status.update(label="Could not process this PDF", state="error", expanded=False)
            except Exception as exc:  # e.g. model download failed (no internet)
                st.session_state.doc_error = (
                    "Something went wrong while processing the PDF. If this is the first run, "
                    f"check your internet connection (the embedding model must download). Details: {exc}"
                )
                status.update(label="Could not process this PDF", state="error", expanded=False)

doc_ready = st.session_state.index is not None

if st.session_state.doc_error:
    st.error(f"⚠️ {st.session_state.doc_error}")
elif doc_ready:
    name = html.escape(st.session_state.doc_name)
    st.markdown(
        f"""
        <div class="doc-pill">
          <span>📄</span><span class="name">{name}</span>
          <span class="stat">{len(st.session_state.pages)} pages with text</span>
          <span class="stat">{len(st.session_state.chunks)} chunks indexed</span>
          <span class="ok">● Ready to answer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chat area
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="card"><h3>2 · Ask your doubts</h3>'
    '<div class="muted">Answers use only the uploaded chapter and show the exact source line.</div></div>',
    unsafe_allow_html=True,
)


def render_source(msg):
    if not msg.get("found") or not msg.get("quote"):
        return
    quote = html.escape(msg["quote"])
    page = msg.get("page")
    label = msg.get("page_label")
    page_text = f"Page {page}"
    if label and str(label) != str(page):
        page_text += f" (printed page {html.escape(str(label))})"
    note = "" if msg.get("quote_verified") else (
        '<span class="note">Closest matching line, auto-selected from the textbook</span>'
    )
    st.markdown(
        f"""
        <div class="source-box">
          <div class="label">📑 Source from textbook</div>
          <blockquote>“{quote}”</blockquote>
          <span class="page">{page_text}</span>{note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message(msg):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(msg["content"])
        return
    with st.chat_message("assistant", avatar="📘"):
        if msg.get("error"):
            st.error(msg["content"])
            return
        if msg.get("kind") == "simpler":
            st.markdown('<span class="simple-tag">✨ Simpler explanation</span>', unsafe_allow_html=True)
        else:
            st.markdown("**📝 Answer**")
        if msg.get("found"):
            st.markdown(msg["content"])
        else:
            st.info(msg["content"])
        render_source(msg)
        if msg.get("sources"):
            with st.expander("🔍 Retrieved textbook chunks (top 3)"):
                for i, chunk in enumerate(msg["sources"], start=1):
                    st.markdown(f"**Chunk {i} · Page {chunk['page']} · similarity {chunk['score']:.2f}**")
                    st.caption(chunk["text"])


if not st.session_state.messages:
    hint = "Ask anything about the chapter you uploaded." if doc_ready else "Upload a PDF above to start."
    st.markdown(
        f'<div class="empty-state"><div class="icon">💬</div>{hint}</div>',
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    render_message(message)


# ---------------------------------------------------------------------------
# Handle a new question
# ---------------------------------------------------------------------------
question = st.chat_input(
    "Type your doubt here, e.g. 'What is photosynthesis?'" if doc_ready else "Upload a PDF first…",
    disabled=not doc_ready,
)

if question and doc_ready:
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.spinner("Searching the textbook and writing your answer…"):
        try:
            retrieved = rag.retrieve_relevant_chunks(
                question, st.session_state.index, st.session_state.chunks, get_embedding_model()
            )
            result = rag.generate_answer(question, retrieved, api_key=current_api_key())
            assistant_msg = {
                "role": "assistant",
                "content": result["answer"],
                "found": result["found"],
                "quote": result["quote"],
                "page": result["page"],
                "page_label": result["page_label"],
                "quote_verified": result["quote_verified"],
                "sources": retrieved,
                "kind": "answer",
            }
        except rag.GeminiError as exc:
            assistant_msg = {"role": "assistant", "content": f"⚠️ {exc}", "error": True}
        except Exception as exc:
            assistant_msg = {
                "role": "assistant",
                "content": f"⚠️ Unexpected error: {exc}. Please try again.",
                "error": True,
            }
    st.session_state.messages.append(assistant_msg)
    st.rerun()


# ---------------------------------------------------------------------------
# Handle "Explain Simpler"
# ---------------------------------------------------------------------------
if st.session_state.pending_simplify:
    st.session_state.pending_simplify = False
    target = last_answer_message()
    if target:
        with st.spinner("Making it simpler…"):
            try:
                simpler = rag.explain_simpler(target["content"], api_key=current_api_key())
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": simpler,
                    "found": True,
                    "quote": target.get("quote"),
                    "page": target.get("page"),
                    "page_label": target.get("page_label"),
                    "quote_verified": target.get("quote_verified"),
                    "kind": "simpler",
                })
            except rag.GeminiError as exc:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ {exc}", "error": True}
                )
    st.rerun()


# ---------------------------------------------------------------------------
# Action buttons
# ---------------------------------------------------------------------------
col1, col2, _ = st.columns([1.2, 1, 3])
with col1:
    if st.button("✨ Explain Simpler", type="primary", disabled=last_answer_message() is None,
                 help="Rewrite the last answer in easier words"):
        st.session_state.pending_simplify = True
        st.rerun()
with col2:
    st.button("🗑️ Clear Chat", on_click=clear_chat, disabled=not st.session_state.messages)
