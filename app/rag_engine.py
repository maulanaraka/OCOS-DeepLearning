from embedder import embed_text
from vector_store import retrieve_similar_chunks
from llm import generate_answer

def query_with_rag(question: str, top_k=3) -> str:
    query_emb = embed_text(question)
    top_chunks = retrieve_similar_chunks(query_emb, top_k=top_k)
    context = "\n".join(top_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return generate_answer(prompt)