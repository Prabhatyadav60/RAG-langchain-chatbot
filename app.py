import os
import io
from textwrap import wrap
import streamlit as st
import pandas as pd
from datetime import datetime
from fpdf import FPDF
from langchain.schema import HumanMessage, AIMessage
from backend import build_or_load_index, query_faiss_index, get_llm_response

try:
    import docx
except ImportError:
    docx = None
try:
    import pptx
except ImportError:
    pptx = None

# Page setup
st.set_page_config(page_title="📘 Document Chat Agent", layout="wide")

# Load API Key
try:
    GROQ_API_KEY = st.secrets["GROQ"]["API_KEY"]
except KeyError:
    st.error("🚫 GROQ API key not found. Please add it under [GROQ] API_KEY in Streamlit secrets.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    uploaded_file = st.file_uploader(
        "📄 Upload your document (PDF, TXT, DOCX, PPTX, CSV)",
        type=["pdf", "txt", "docx", "pptx", "csv"]
    )
    show_context = st.checkbox("🔎 Show Retrieved Contexts")
    if st.button("🧹 Clear Conversation"):
        st.session_state.pop("chat_history", None)

    if st.button("🖨️ Download as PDF") and "chat_history" in st.session_state:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)

        pdf.set_fill_color(230, 230, 250)  # light lavender for header
        pdf.cell(0, 10, "Chat Transcript", ln=True, align="C", fill=True)
        pdf.ln(4)

        for msg in st.session_state.chat_history:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content.replace("\n", " ")

            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"{role}:", ln=True)

            pdf.set_font("Arial", "", 12)
            for line in wrap(content, width=100):
                pdf.multi_cell(0, 10, line)
            pdf.ln(2)

        # handle both bytearray and str output
        raw = pdf.output(dest='S')
        if isinstance(raw, bytearray):
            pdf_bytes = bytes(raw)
        else:
            pdf_bytes = raw.encode('latin-1')

        buf = io.BytesIO(pdf_bytes)
        st.download_button(
            label="📥 Download Chat as PDF",
            data=buf,
            file_name="chat_history.pdf",
            mime="application/pdf"
        )

# Main App Title
st.markdown(
    "<h1 style='text-align:center;color:#4A90E2;'>💬 Conversational Document Chat Agent</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Upload documents and chat with their contents via an LLM-powered RAG pipeline.</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# Text extraction helper
def extract_text(path, ftype):
    if ftype == 'txt':
        return open(path, 'r', encoding='utf-8').read()
    if ftype == 'csv':
        return pd.read_csv(path).to_string()
    if ftype == 'docx' and docx:
        doc = docx.Document(path)
        return '\n'.join(p.text for p in doc.paragraphs)
    if ftype == 'pptx' and pptx:
        prs = pptx.Presentation(path)
        return '\n'.join(
            shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, 'text')
        )
    return None

# Display retrieved contexts
def display_contexts(ctxs):
    st.markdown("### 🔍 Retrieved Contexts")
    for i, c in enumerate(ctxs, 1):
        st.markdown(
            f"<div style='background:#f0f2f6;padding:10px;border-radius:8px;margin:4px 0;'>"
            f"<strong>Context {i}:</strong> {c}</div>",
            unsafe_allow_html=True
        )

# Main upload and chat logic
if uploaded_file:
    os.makedirs('docs', exist_ok=True)
    path = os.path.join('docs', uploaded_file.name)
    with open(path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # determine extension
    ext = uploaded_file.name.rsplit('.', 1)[1].lower()

    # build or load index via your backend
    @st.cache_resource
    def load_index(p):
        return build_or_load_index(p)

    # for non-PDFs, extract text to a .txt first
    if ext in ['txt', 'csv', 'docx', 'pptx']:
        raw = extract_text(path, ext)
        txt_path = os.path.splitext(path)[0] + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as tf:
            tf.write(raw)
        index_path = txt_path
    else:
        index_path = path

    embedder, texts, index = load_index(index_path)
    st.session_state.setdefault('chat_history', [])

    # replay previous messages
    for m in st.session_state.chat_history:
        role = 'user' if isinstance(m, HumanMessage) else 'assistant'
        with st.chat_message(role):
            st.markdown(m.content)

    # new user query
    if q := st.chat_input('💬 Ask something...'):
        st.session_state.chat_history.append(HumanMessage(content=q))
        with st.chat_message('user'):
            st.markdown(q)

        ctxs = query_faiss_index(q, embedder, index, texts)
        if show_context:
            display_contexts(ctxs)

        resp = get_llm_response(
            api_key=GROQ_API_KEY,
            query=q,
            contexts=ctxs,
            chat_history=st.session_state.chat_history
        )
        st.session_state.chat_history.append(AIMessage(content=resp))
        with st.chat_message('assistant'):
            st.markdown(resp)
else:
    st.info('📂 Upload a document to begin.')
