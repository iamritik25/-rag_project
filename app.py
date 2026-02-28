import os
import time
import random
import requests
import streamlit as st
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------------- CONFIG ----------------

MODEL_NAME = "mistral"

UPLOAD_DIR = "pdfs"
INDEX_DIR = "indexes"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ---------------- CLEANUP ----------------
def cleanup_session_data():
    for f in os.listdir(INDEX_DIR):
        os.remove(os.path.join(INDEX_DIR, f))
    for f in os.listdir(UPLOAD_DIR):
        os.remove(os.path.join(UPLOAD_DIR, f))

# ---------------- EMBEDDING MODEL ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
# ---------------- RERANKER ----------------
@st.cache_resource
def load_reranker():
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

rerank_tokenizer, rerank_model = load_reranker()

# ---------------- PDF UTIL ----------------
def read_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ---------------- FAISS ----------------
def build_faiss(chunks, pdf_name):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, f"{INDEX_DIR}/{pdf_name}.index")

    with open(f"{INDEX_DIR}/{pdf_name}.txt", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.replace("\n", " ") + "\n")

def load_faiss(pdf_name):
    index = faiss.read_index(f"{INDEX_DIR}/{pdf_name}.index")
    with open(f"{INDEX_DIR}/{pdf_name}.txt", "r", encoding="utf-8") as f:
        chunks = f.readlines()
    return index, chunks

# ---------------- QUERY ----------------
def ask_ollama(context, question):
    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not present, say "Not found in the document".

Context:
{context}

Question:
{question}
"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload)
    answer = r.json()["response"].strip()

    # 🔒 HARD ENFORCEMENT GUARD
    if answer.startswith("Not found in the document"):
        return "Not found in the document."

    return answer
    

def rerank_chunks(question, chunks, top_k=3):
    pairs = [(question, chunk) for chunk in chunks]

    inputs = rerank_tokenizer(
        [q for q, c in pairs],
        [c for q, c in pairs],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        scores = rerank_model(**inputs).logits.squeeze()

    ranked_indices = torch.argsort(scores, descending=True)

    return [chunks[i] for i in ranked_indices[:top_k]]


def query_pdf(pdf_name, question):
    index, chunks = load_faiss(pdf_name)
    q_embedding = embedder.encode([question])

    # Retrieve more candidates
    _, ids = index.search(np.array(q_embedding), k=5)

    retrieved_chunks = [chunks[i] for i in ids[0]]

    # Rerank
    reranked_chunks = rerank_chunks(question, retrieved_chunks, top_k=3)

    context = "".join(reranked_chunks)

    return ask_ollama(context, question)

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="👻 Agentic PDF AI", layout="wide")

# ---------------- SESSION STATE ----------------
if "ghost_mood" not in st.session_state:
    st.session_state.ghost_mood = "😄"

if "fail_count" not in st.session_state:
    st.session_state.fail_count = 0
    
if "active" not in st.session_state:
    st.session_state.active = True

# ---------------- CSS ----------------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0c0c0c, #000);
    color: white;
}

.neon {
    color: #ff9f1c;
    text-shadow: 0 0 12px #ff9f1c, 0 0 30px #ff6f00;
}

.neon-red {
    color: #ff4d4d;
    text-shadow: 0 0 12px red, 0 0 30px darkred;
}

.glass {
    background: rgba(20,20,20,0.85);
    border-radius: 18px;
    padding: 24px;
    border: 1px solid rgba(255,159,28,0.3);
    box-shadow: 0 0 25px rgba(255,159,28,0.25);
}

.stButton>button {
    background: linear-gradient(135deg, #ff9f1c, #ff6a00);
    color: black;
    font-weight: 700;
    border-radius: 14px;
    box-shadow: 0 0 25px rgba(255,159,28,0.9);
}

.ghost {
    position: fixed;
    font-size: 40px;
    animation: float 18s linear infinite;
    opacity: 0.9;
}

.shake {
    animation: shake 0.35s infinite;
}

.pulse-red {
    animation: pulse 1s infinite;
}

@keyframes float {
    from { transform: translateY(100vh); }
    to { transform: translateY(-120vh); }
}

@keyframes shake {
    0% { transform: translateX(0); }
    25% { transform: translateX(-6px); }
    50% { transform: translateX(6px); }
    75% { transform: translateX(-6px); }
    100% { transform: translateX(0); }
}

@keyframes pulse {
    0% { box-shadow: 0 0 10px red; }
    50% { box-shadow: 0 0 35px red; }
    100% { box-shadow: 0 0 10px red; }
}

.ghost1 { left: 5%; animation-duration: 22s; }
.ghost2 { left: 45%; animation-duration: 18s; }
.ghost3 { left: 75%; animation-duration: 26s; }
</style>

<div class="ghost ghost1">👻</div>
<div class="ghost ghost2">👻</div>
<div class="ghost ghost3">👻</div>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(f"""
<h1 class="neon">👻 Agentic RAG Document Q&A System {st.session_state.ghost_mood}</h1>
<p>Local AI PDF Chat • Hallucination Guard Enabled</p>
""", unsafe_allow_html=True)

# ---------------- END SESSION ----------------
if st.button("🛑 End Session"):
    cleanup_session_data()
    st.session_state.active = False
    st.session_state.ghost_mood = "☠️"
    st.markdown("<h1 class='neon-red'>Session Terminated</h1>", unsafe_allow_html=True)
    st.stop()

# ---------------- STOP IF ENDED ----------------
if not st.session_state.active:
    st.stop()



# ---------------- UPLOAD ----------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
uploaded_files = st.file_uploader("📤 Upload PDFs", type="pdf", accept_multiple_files=True)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files:
    for file in uploaded_files:
        path = f"{UPLOAD_DIR}/{file.name}"
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            with st.spinner(f"Indexing {file.name}..."):
                text = read_pdf_text(path)
                chunks = chunk_text(text)
                build_faiss(chunks, file.name)
    st.success("✅ PDFs indexed successfully")

# ---------------- SELECT + ASK ----------------
indexed_pdfs = [f.replace(".index", "") for f in os.listdir(INDEX_DIR) if f.endswith(".index")]

if indexed_pdfs:
    selected_pdf = st.selectbox("📄 Choose a PDF", indexed_pdfs)
    question = st.text_input("❓ Ask a question", placeholder="Explain this document...")

    if st.button("👻 Ask") and question.strip():
        st.session_state.ghost_mood = "🤔"

        with st.spinner("👻 Ghost is thinking..."):
            answer = query_pdf(selected_pdf, question)

        # ---- Hallucination detection ----
        if "Not found in the document" in answer:
            st.session_state.fail_count += 1

            if st.session_state.fail_count >= 3:
                st.session_state.ghost_mood = "😡"
            else:
                st.session_state.ghost_mood = "😵"

            st.markdown("""
            <div class="ghost shake pulse-red" style="
                left: 50%;
                top: 40%;
                font-size: 90px;
                z-index: 9999;
            ">👻</div>
            """, unsafe_allow_html=True)
        else:
            st.session_state.fail_count = 0
            st.session_state.ghost_mood = "😄"

        # ---- Typing animation ----
        st.markdown("### 👻 Answer")
        placeholder = st.empty()
        typed = ""
        for ch in answer:
            typed += ch
            placeholder.markdown(typed)
            time.sleep(0.015)
else:
    st.info("Upload PDFs to begin.")
