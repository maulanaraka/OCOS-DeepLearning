from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import numpy as np
import google.generativeai as genai

# --- Konfigurasi Gemini ---
genai.configure(api_key="AIzaSyBGfprRY9PW9PED_C_XcqPAZwHatxfMkbo")
embedder = genai.embedder.Embedder(model_name="models/embedding-001")
llm = genai.GenerativeModel('gemini-1.5-flash')

# --- Load FAISS index dan mapping teks ---
faiss_index = faiss.read_index("bps_faiss.index")
with open("bps_faiss_texts.txt", encoding="utf-8") as f:
    texts = [line.strip() for line in f.readlines()]

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

class QueryRequest(BaseModel):
    question: str
    top_n: int = 5

@app.post("/query")
async def query_semantic(req: QueryRequest):
    # 1. Embed pertanyaan
    emb = embedder.embed(content=req.question)["embedding"]
    emb = np.array(emb, dtype="float32").reshape(1, -1)

    # 2. Cari top-N di FAISS
    D, I = faiss_index.search(emb, req.top_n)
    context_chunks = [texts[i] for i in I[0]]

    # 3. Gabungkan context dan pertanyaan, kirim ke LLM
    context_str = "\n".join(context_chunks)
    prompt = f"Context:\n{context_str}\n\nQuestion: {req.question}\nAnswer in Bahasa Indonesia:"
    response = llm.generate_content(prompt)
    answer = response.text

    return {
        "answer": answer,
        "context": context_chunks
    }