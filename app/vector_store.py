import faiss
import numpy as np

# Load FAISS index
faiss_index = faiss.read_index("faiss_index.idx")

# Load original texts
with open("faiss_texts.txt", "r", encoding="utf-8") as f:
    text_chunks = [line.strip() for line in f.readlines()]

def retrieve_similar_chunks(query_embedding, top_k=3):
    query_vector = np.array([query_embedding]).astype("float32")
    distances, indices = faiss_index.search(query_vector, top_k)
    return [text_chunks[i] for i in indices[0] if i < len(text_chunks)]