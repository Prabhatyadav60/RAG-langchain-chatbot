import os
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from backend import build_or_load_index, query_faiss_index, get_llm_response
from streamlit.errors import StreamlitAPIException, StreamlitSecretNotFoundError

# Page setup
st.set_page_config(page_title="📘 PDF Chat Agent", layout="wide")

# Load API Key from Streamlit secrets or environment variable
GROQ_API_KEY = None
try:
    GROQ_API_KEY = st.secrets["GROQ"]["API_KEY"]
except (KeyError, StreamlitAPIException, StreamlitSecretNotFoundError):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "🚫 GROQ API key not found.\n"
        "• On Streamlit Cloud: go to 'Manage App' → 'Secrets' and add:\n"
        "  [GROQ]\n  API_KEY = \"your_key_here\"\n"
        "• Or set the environment variable: GROQ_API_KEY"
    )
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    uploaded_pdf = st.file_uploader("📄 Upload your PDF", type=["pdf"])
    show_context = st.checkbox("🔎 Show Retrieved Contexts")
    if st.button("🧹 Clear Conversation"):
        st.session_state.pop("chat_history", None)

# Main App
st.markdown(
    "<h1 style='text-align: center; color: #4A90E2;'>💬 Conversational PDF Chat Agent</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Ask questions from your PDF using LLM and get intelligent responses!</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# Display retrieved contexts
def display_contexts(contexts):
    st.markdown("### 🔍 Retrieved Contexts")
    for i, ctx in enumerate(contexts, 1):
        st.markdown(
            f"<div style='background-color:#f0f2f6;padding:10px;border-radius:10px;margin:5px 0;'><strong>Context {i}:</strong> {ctx}</div>",
            unsafe_allow_html=True
        )

# Chat interface after upload
if uploaded_pdf:
    os.makedirs("pdfs", exist_ok=True)
    pdf_path = os.path.join("pdfs", uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    @st.cache_resource
    def load_index(path):
        return build_or_load_index(path)

    embedder, texts, index = load_index(pdf_path)
    st.session_state.setdefault("chat_history", [])

    # Render previous messages
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

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
