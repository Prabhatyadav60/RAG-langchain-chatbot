import os
import io
import time
from textwrap import wrap
import streamlit as st
import pandas as pd
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

# ——— Simple PDF class with time module for header ———
class SimplePDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Chat Conversation Report", ln=True, align="C")
        # Use time.strftime for a consistent local timestamp
        timestamp = time.strftime("%B %d, %Y %H:%M:%S", time.localtime())
        self.set_font("Arial", "", 10)
        self.cell(0, 8, timestamp, ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def make_pdf_bytes(chat_history):
    pdf = SimplePDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "", 12)

    for msg in chat_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        text = msg.content.strip().replace("\n", " ")
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"{role}:", ln=True)
        pdf.set_font("Arial", "", 12)
        for line in wrap(text, width=90):
            pdf.cell(5)  # indent
            pdf.cell(0, 8, line, ln=True)
        pdf.ln(2)

    raw = pdf.output(dest="S")
    return bytes(raw) if isinstance(raw, bytearray) else raw.encode("latin-1")


# ——— Streamlit UI ———
st.set_page_config(page_title="📘 Document Chat Agent", layout="wide")
try:
    GROQ_API_KEY = st.secrets["GROQ"]["API_KEY"]
except KeyError:
    st.error("🚫 Add your GROQ API key under [GROQ] API_KEY in secrets.")
    st.stop()

st.sidebar.markdown("## ⚙️ Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload document (PDF, TXT, DOCX, PPTX, CSV)",
    type=["pdf", "txt", "docx", "pptx", "csv"]
)
show_context = st.sidebar.checkbox("🔎 Show Retrieved Contexts")
if st.sidebar.button("🧹 Clear Conversation"):
    st.session_state.pop("chat_history", None)

st.markdown("<h1 style='text-align:center;'>💬 Document Chat Agent</h1>",
            unsafe_allow_html=True)
st.markdown("---")

def extract_text(path, ftype):
    if ftype == "txt":
        return open(path, "r", encoding="utf-8").read()
    if ftype == "csv":
        return pd.read_csv(path).to_string()
    if ftype == "docx" and docx:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    if ftype == "pptx" and pptx:
        prs = pptx.Presentation(path)
        return "\n".join(
            shp.text for sl in prs.slides for shp in sl.shapes if hasattr(shp, "text")
        )
    return ""

def display_contexts(ctxs):
    st.markdown("### 🔍 Retrieved Contexts")
    for i, c in enumerate(ctxs, 1):
        st.markdown(f"- **Context {i}:** {c}")

if uploaded_file:
    os.makedirs("docs", exist_ok=True)
    path = os.path.join("docs", uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    ext = uploaded_file.name.rsplit(".", 1)[1].lower()

    @st.cache_resource
    def load_idx(p): return build_or_load_index(p)

    if ext in ["txt", "csv", "docx", "pptx"]:
        raw = extract_text(path, ext)
        txt_path = path + ".txt"
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(raw)
        idx_path = txt_path
    else:
        idx_path = path

    embedder, texts, index = load_idx(idx_path)
    st.session_state.setdefault("chat_history", [])

    for m in st.session_state.chat_history:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(m.content)

    if q := st.chat_input("💬 Ask something..."):
        st.session_state.chat_history.append(HumanMessage(content=q))
        with st.chat_message("user"):
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
        with st.chat_message("assistant"):
            st.markdown(resp)

    if st.session_state.chat_history:
        pdf_bytes = make_pdf_bytes(st.session_state.chat_history)
        st.sidebar.download_button(
            "📥 Download Chat as PDF",
            data=pdf_bytes,
            file_name="chat_report.pdf",
            mime="application/pdf"
        )
else:
    st.info("📂 Upload a document to begin.")
