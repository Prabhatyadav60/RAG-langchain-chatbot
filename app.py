import os
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from backend import build_or_load_index, query_faiss_index, get_llm_response

# Page setup
st.set_page_config(page_title="📘 PDF Chat Agent", layout="wide")

# Load API Key
try:
    GROQ_API_KEY = st.secrets["GROQ"]["API_KEY"]
except KeyError:
    st.error("🚫 GROQ API key not found in Streamlit secrets. Please add it under [GROQ] API_KEY.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    uploaded_pdf = st.file_uploader("📄 Upload your PDF", type=["pdf"])
    show_context = st.checkbox("🔎 Show Retrieved Contexts")
    if st.button("🧹 Clear Conversation"):
        st.session_state.chat_history = []

# Main App
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>💬 Conversational PDF Chat Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask questions from your PDF using LLM and get intelligent responses!</p>", unsafe_allow_html=True)
st.markdown("---")

# Chat starts after PDF is uploaded
if uploaded_pdf:
    os.makedirs("pdfs", exist_ok=True)
    pdf_path = os.path.join("pdfs", uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    @st.cache_resource
    def load_index(path):
        return build_or_load_index(path)

    embedder, texts, index = load_index(pdf_path)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display contexts if enabled
    def display_contexts(contexts):
        st.markdown("### 🔍 Retrieved Contexts")
        for i, ctx in enumerate(contexts, 1):
            st.markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:10px;margin:5px 0;'><strong>Context {i}:</strong> {ctx}</div>", unsafe_allow_html=True)

    # Show chat history
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # Chat input
    if prompt := st.chat_input("💬 Ask something from the PDF..."):
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        contexts = query_faiss_index(prompt, embedder, index, texts)
        if show_context:
            display_contexts(contexts)

        answer = get_llm_response(
            api_key=GROQ_API_KEY,
            query=prompt,
            contexts=contexts,
            chat_history=st.session_state.chat_history
        )

        st.session_state.chat_history.append(AIMessage(content=answer))
        with st.chat_message("assistant"):
            st.markdown(answer)

else:
    st.info("📂 Upload a PDF from the sidebar to get started.")
