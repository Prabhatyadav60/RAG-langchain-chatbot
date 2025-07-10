import os
from dotenv import load_dotenv
import numpy as np
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
load_dotenv() 
PDF_PATH    = "pdfs/fine_tuning.pdf"
VECTOR_DIR  = "vector_store"
EMB_PATH    = os.path.join(VECTOR_DIR, "embeddings.npy")
IDX_PATH    = os.path.join(VECTOR_DIR, "faiss_index.idx")

os.makedirs(VECTOR_DIR, exist_ok=True)
need_build = True
if os.path.exists(EMB_PATH) and os.path.exists(IDX_PATH):
    try:
        print("🔄 Loading embeddings & FAISS index from disk...")
        embeddings = np.load(EMB_PATH)
        index      = faiss.read_index(IDX_PATH)
        embedder   = SentenceTransformer("all-MiniLM-L6-v2")
        need_build = False
    except Exception as e:
        print("⚠️ Failed to load saved index:", e)
        print("→ Will rebuild embeddings & index.")

if need_build:
    print("🚧 Building embeddings & FAISS index (this runs once)...")
    loader   = PyPDFLoader(PDF_PATH)
    pages    = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 500,
        chunk_overlap = 50,
        separators    = ["\n\n", "\n", " ", ""]
    )
    chunks   = splitter.split_documents(pages)
    texts    = [chunk.page_content for chunk in chunks]
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    np.save(EMB_PATH, embeddings)
    faiss.write_index(index, IDX_PATH)
    print(f"✅ Saved embeddings → {EMB_PATH}")


groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("🔑 Please set your GROQ_API_KEY in .env")

llm = ChatGroq(
    model_name = "llama3-8b-8192", 
    api_key    = groq_api_key      
)


print("\n🤖 PDF QA Agent ready! Type 'exit' to quit.\n")

while True:
    query = input("📝 Your question: ").strip()
    if query.lower() in ("exit", "quit"):
        print("👋 Goodbye!")
        break
    if not query:
        continue
    q_emb = embedder.encode([query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k=3)

    if 'texts' not in locals():
        loader = PyPDFLoader(PDF_PATH)
        pages  = loader.load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size    = 500,
            chunk_overlap = 50
        ).split_documents(pages)
        texts  = [chunk.page_content for chunk in chunks]

    contexts = [texts[i] for i in I[0]]

    system_msg = HumanMessage(
        content=(
            "You are a helpful assistant. "
            "If the user’s question relates to the provided context, use that. "
            "Otherwise, feel free to answer generally as a normal AI assistant."
        )
    )

    user_content = "Use the following extracted context to answer the question.\n\n"
    for idx, ctx in enumerate(contexts, 1):
        user_content += f"Context {idx}:\n{ctx}\n\n"
    user_content += f"Question: {query}\nAnswer:"

    user_msg = HumanMessage(content=user_content)
    response = llm.invoke([system_msg, user_msg])

    print("\n💡 Answer:\n")
    print(response.content)
    print("\n" + "—" * 40 + "\n")
