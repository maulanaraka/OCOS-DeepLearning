from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from threading import Lock

app = Flask(__name__)
CORS(app)  # Untuk testing lokal



# --- Konfigurasi Kunci API Gemini ---
try:
    api_key = "AIzaSyBGfprRY9PW9PED_C_XcqPAZwHatxfMkbo" # GANTI DENGAN KUNCI API ANDA
    if not api_key:
        raise ValueError("Kunci API Gemini tidak ditemukan. Harap atur dalam kode atau sebagai variabel lingkungan.")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error saat menginisialisasi klien Gemini: {e}")
    gemini_model = None

# --- Load FAISS index dan mapping teks ---
faiss_index = faiss.read_index("faiss_index.idx")
with open("bps_faiss_texts.txt", encoding="utf-8") as f:
    texts = [line.strip() for line in f.readlines()]

# --- Load SentenceTransformer sekali saja di awal ---
embedder = None
embedder_lock = Lock()
def get_embedder():
    global embedder
    with embedder_lock:
        if embedder is None:
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return embedder

@app.route("/")
def index():
    """Menyajikan antarmuka pengguna utama dari file main.html."""
    return render_template('main.html')

@app.route("/embed", methods=["POST"])
def embed_text():
    """Endpoint khusus untuk embed teks (bukan untuk query/ask)."""
    data = request.get_json()
    texts = data.get("texts", [])
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Input harus berupa list teks."}), 400
    embedder = get_embedder()
    embeddings = embedder.encode(texts).tolist()
    return jsonify({"embeddings": embeddings})

@app.route("/ask", methods=["POST"])
def ask():
    """Menerima pertanyaan, mengirimkannya ke Gemini AI, dan mengembalikan jawabannya."""
    if not gemini_model:
        return jsonify({"answer": "Error: Klien Gemini AI tidak terinisialisasi. Periksa kunci API Anda."}), 500
    data = request.get_json()
    question = data.get("question", "")
    top_n = int(data.get("top_n", 5))

    if not question:
        return jsonify({"answer": "Tidak ada pertanyaan yang diberikan."}), 400

    embedder = get_embedder()
    # 1. Embed pertanyaan
    emb = embedder.encode([question]).astype('float32')

    # 2. Cari top-N di FAISS
    D, I = faiss_index.search(emb, top_n)
    context_chunks = [texts[i] for i in I[0]]

    # 3. Gabungkan context dan pertanyaan, kirim ke LLM
    context_str = "\n".join(context_chunks)
    prompt = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer in Bahasa Indonesia:"
    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"Terjadi kesalahan saat menghubungi Gemini API: {str(e)}"
        return jsonify({"answer": answer}), 500

    return jsonify({
        "answer": answer
    })

if __name__ == "__main__":
    # Menjalankan aplikasi Flask dalam mode debug di port 5000
    app.run(debug=True, port=5000)