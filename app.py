

import os
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from backend import build_or_load_index, query_faiss_index, get_llm_response
from fpdf import FPDF
import pandas as pd
import io
try:
    import docx
except ImportError:
    docx = None
try:
    import pptx
except ImportError:
    pptx = None
from datetime import datetime


def create_pdf():
    class ReportPDF(FPDF):
        def header(self):
       
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'Chat Conversation Report', ln=True, align='C')
            # Date
            self.set_font('Arial', '', 10)
            self.cell(0, 8, datetime.now().strftime('%B %d, %Y %H:%M'), ln=True, align='C')
            self.ln(5)
            # Horizontal line
            self.set_draw_color(0, 0, 0)
            self.set_line_width(0.5)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)

        def footer(self):
            # Page number
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', align='C')

    return ReportPDF()


st.set_page_config(page_title="📘 Document Chat Agent", layout="wide")


try:
    GROQ_API_KEY = st.secrets["GROQ"]["API_KEY"]
except KeyError:
    st.error("🚫 GROQ API key not found. Please add it under [GROQ] API_KEY in Streamlit secrets.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    uploaded_file = st.file_uploader(
        "📄 Upload your document (PDF)",
        type=["pdf"]
    )
    show_context = st.checkbox("🔎 Show Retrieved Contexts")
    if st.button("🧹 Clear Conversation"):
        st.session_state.pop("chat_history", None)
    if st.button("🖨️ Download Conversation as PDF") and "chat_history" in st.session_state:
      
        pdf = create_pdf()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', '', 12)
        # Table header
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(40, 10, 'Role', border=1, fill=True)
        pdf.cell(0, 10, 'Message', border=1, fill=True, ln=True)
        pdf.set_fill_color(245, 245, 245)
        # Add conversation rows
        fill = False
        for msg in st.session_state.chat_history:
            role = 'User' if isinstance(msg, HumanMessage) else 'Assistant'
            content = msg.content.replace('\n', ' ')
            pdf.cell(40, 10, role, border=1, fill=fill)
            pdf.multi_cell(0, 10, content, border=1, fill=fill)
            fill = not fill
    
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        buf = io.BytesIO(pdf_bytes)
        st.download_button(
            label="Download Chat Report",
            data=buf,
            file_name="chat_conversation_report.pdf",
            mime="application/pdf"
        )


st.markdown(
    "<h1 style='text-align:center;color:#4A90E2;'>💬 Conversational Document Chat Agent</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Upload documents and chat with their contents via an LLM-powered RAG pipeline.</p>",
    unsafe_allow_html=True
)
st.markdown("---")

def extract_text(path, ftype):
    if ftype == 'txt': return open(path, 'r', encoding='utf-8').read()
    if ftype == 'csv': return pd.read_csv(path).to_string()
    if ftype == 'docx' and docx:
        doc = docx.Document(path)
        return '\n'.join(p.text for p in doc.paragraphs)
    if ftype == 'pptx' and pptx:
        prs = pptx.Presentation(path)
        return '\n'.join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, 'text'))
    return None

# Display contexts
def display_contexts(ctxs):
    st.markdown("### 🔍 Retrieved Contexts")
    for i, c in enumerate(ctxs, 1):
        st.markdown(f"<div style='background:#f0f2f6;padding:10px;border-radius:8px;margin:4px 0;'>" +
                    f"<strong>Context {i}:</strong> {c}</div>", unsafe_allow_html=True)


if uploaded_file:
    os.makedirs('docs', exist_ok=True)
    path = os.path.join('docs', uploaded_file.name)
    with open(path, 'wb') as f: f.write(uploaded_file.getbuffer())
    ext = uploaded_file.type.split('/')[-1]

    @st.cache_resource
    def load_index(p): return build_or_load_index(p)

    if ext in ['txt','csv','docx','pptx']:
        raw = extract_text(path, ext)
        txt_path = os.path.splitext(path)[0] + '.txt'
        with open(txt_path,'w',encoding='utf-8') as tf: tf.write(raw)
        index_path = txt_path
    else:
        index_path = path

    embedder, texts, index = load_index(index_path)
    st.session_state.setdefault('chat_history', [])

    
    for m in st.session_state.chat_history:
        role = 'user' if isinstance(m, HumanMessage) else 'assistant'
        with st.chat_message(role): st.markdown(m.content)


    if q := st.chat_input('💬 Ask something...'):
        st.session_state.chat_history.append(HumanMessage(content=q))
        with st.chat_message('user'): st.markdown(q)
        ctxs = query_faiss_index(q, embedder, index, texts)
        if show_context: display_contexts(ctxs)
        resp = get_llm_response(api_key=GROQ_API_KEY, query=q, contexts=ctxs, chat_history=st.session_state.chat_history)
        st.session_state.chat_history.append(AIMessage(content=resp))
        with st.chat_message('assistant'): st.markdown(resp)
else:
    st.info('📂 Upload a document to begin.')
