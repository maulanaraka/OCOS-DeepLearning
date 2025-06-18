from fastapi import FastAPI, Request
from pydantic import BaseModel
from flask_cors import CORS
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import os

# Load FAISS index
faiss_index_path = 'faiss_index.idx'
assert os.path.exists(faiss_index_path), 'FAISS index file not found.'
index = faiss.read_index(faiss_index_path)

# Load embedding model (pastikan sama dengan yang dipakai saat indexing)
try:
    model = SentenceTransformer('indobenchmark/indobert-base-p1')
except Exception:
    model = SentenceTransformer('all-MiniLM-L6-v2')

# Dummy chunks (replace with your actual text chunks)
# Ideally, simpan juga list of all_texts saat indexing, lalu load di sini
import pickle
if os.path.exists('all_texts.pkl'):
    with open('all_texts.pkl', 'rb') as f:
        all_texts = pickle.load(f)
else:
    all_texts = [f'Chunk {i}' for i in range(index.ntotal)]

app = FastAPI()
CORS(app)

class QueryRequest(BaseModel):
    question: str
    top_n: int = 5

@app.post('/query')
def query_semantic_similarity(req: QueryRequest):
    # Embed pertanyaan
    q_emb = model.encode([req.question]).astype('float32')
    # Cari top-N
    D, I = index.search(q_emb, req.top_n)
    # Ambil chunk paling mirip
    results = [all_texts[i] for i in I[0]]
    # Kirim ke LLM (dummy response)
    # Ganti dengan pemanggilan OpenAI/Gemini jika perlu
    context = '\n'.join(results)
    answer = f"[Dummy LLM] Context:\n{context}\n\nJawaban untuk: {req.question}"
    return {"answer": answer, "context": results}

# Jalankan dengan: uvicorn fastapi_app:app --reload
