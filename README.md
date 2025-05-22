# 💬 LangChain RAG-based Conversational Document Chatbot

> **User:** What is this project about?  
> **Assistant:** It’s a Retrieval-Augmented Generation chatbot built with LangChain, FAISS, and a hosted LLM (GROQ or OpenAI). You upload documents (PDF, TXT, DOCX, PPTX, CSV), and it retrieves relevant chunks to answer your questions—right inside a Streamlit app.

---

## 🚀 Features

> **User:** What can it do?  
> **Assistant:**  
> - 🔎 **RAG Search:** Uses FAISS to index your documents and pull the most relevant passages.  
> - 🧠 **LLM Chat:** Leverages GROQ/OpenAI embeddings and completion endpoints for intelligent answers.  
> - 📄 **Supports PDF** 
> - 🌐 **Streamlit UI:** A sleek, interactive chat interface.  
> - 📥 **PDF Export:** Download the full transcript as a neatly formatted PDF with timestamps.  
> - ⚙️ **Modular Backend:** Easy to extend—indexing, embedding, and response logic separated into `backend/`.

---

## 📁 Structure

Langchain-RAG-chatbot/
│
├── backend/
│ ├── embedder.py # build embeddings & FAISS index
│ ├── indexer.py # index loading & saving
│ └── llm_response.py # RAG orchestration & LLM calls
│
├── streamlit_app/
│ ├── app.py # main Streamlit UI + PDF export
│ └── utils.py # text-extraction helpers
│
├── docs/ # auto-generated uploads & temp files
├── requirements.txt # Python dependencies
└── README.md # this file




---

## 🛠 Installation

> **User:** How do I set it up?  
> **Assistant:**  
> ```bash
> git clone https://github.com/yourusername/Langchain-RAG-chatbot.git
> cd Langchain-RAG-chatbot
> python -m venv venv
> source venv/bin/activate        # macOS/Linux
> venv\Scripts\activate           # Windows
> pip install -r requirements.txt
> ```
>
> **User:** What about API keys?  
> **Assistant:**  
> Create `.streamlit/secrets.toml`:
> ```toml
> [GROQ]
> API_KEY = "your-groq-api-key"
>
> [OPENAI]
> API_KEY = "your-openai-api-key"    # if using OpenAI embeddings/completions
> ```

---

## ▶️ Run the App

> **Assistant:**  
> ```bash
> streamlit run streamlit_app/app.py
> ```

---

## 💬 Usage Examples

> **User:** “Summarize key findings of this report.”  
> **Assistant:** Returns a concise summary.  
>  
> **User:** “List deadlines mentioned.”  
> **Assistant:** Extracts all dates.  
>  
> **User:** “Translate this paragraph to plain English.”  
> **Assistant:** Provides a simplified version.

---

## 🌐 Supported File Types

> **Assistant:**  
> - PDF (`.pdf`)  


---

## 🧰 Tech Stack

> **Assistant:**  
> - **LangChain** – RAG orchestration  
> - **FAISS** – Vector search  
> - **GROQ/OpenAI** – Embeddings & LLM  
> - **Streamlit** – Frontend  
> - **FPDF2** – PDF export  
> - **python-docx**, **python-pptx**, **pandas** – file parsing

---

## 🔮 Future Enhancements

> **User:** What’s next?  
> **Assistant:**  
> - Persistent chat memory  
> - URL & YouTube transcript ingestion  
> - Multilingual support  
> - User authentication & profiles

---



---


