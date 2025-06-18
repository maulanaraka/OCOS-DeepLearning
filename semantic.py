# semantic_rag_system.py
# Semantic Similarity-based RAG QA System for BPS Indonesia Data

import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from typing import List, Dict, Any
import asyncio

# Initialize FastAPI app
app = FastAPI(title="BPS Indonesia Semantic RAG System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENAI_API_KEY = "your-openai-api-key-here"  # Replace with actual key
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB_PATH = "faiss_index.pkl"
CHUNKS_PATH = "text_chunks.pkl"

# Initialize models
openai.api_key = OPENAI_API_KEY
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Global variables for vector database
faiss_index = None
text_chunks = []
chunk_metadata = []

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    relevant_chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class DocumentProcessor:
    """Process BPS documents and create text chunks"""
    
    def __init__(self):
        self.chunks = []
        self.metadata = []
    
    def load_bps_data(self, data_path: str = "bps_sample_data.json"):
        """Load BPS data from JSON file"""
        bps_data = {
            "population_data": [
                {
                    "id": "pop_001",
                    "province": "DKI Jakarta",
                    "description": "DKI Jakarta sebagai ibu kota Indonesia memiliki jumlah penduduk 10.679.000 jiwa pada tahun 2024 dengan laju pertumbuhan 1,11%. Kepadatan penduduk mencapai 16.030 jiwa per km2, menjadikannya daerah terpadat di Indonesia. Komposisi penduduk terdiri dari 50,8% laki-laki dan 49,2% perempuan."
                },
                {
                    "id": "pop_002", 
                    "province": "Jawa Barat",
                    "description": "Jawa Barat merupakan provinsi dengan populasi terbesar di Indonesia dengan 48.939.000 jiwa pada 2024. Pertumbuhan penduduk relatif stabil di 0,53% per tahun. Kepadatan penduduk mencapai 1.386 jiwa per km2 dengan distribusi gender yang hampir seimbang."
                },
                {
                    "id": "pop_003",
                    "province": "Jawa Timur", 
                    "description": "Jawa Timur memiliki 39.871.000 penduduk pada 2024 dengan pertumbuhan yang lambat 0,35% per tahun. Kepadatan penduduk 840 jiwa per km2. Provinsi ini memiliki proporsi perempuan sedikit lebih tinggi yaitu 50,2%."
                }
            ],
            "labor_data": [
                {
                    "id": "labor_001",
                    "province": "DKI Jakarta",
                    "description": "Angkatan kerja DKI Jakarta pada Februari 2024 sebesar 5,43 juta orang dengan tingkat pengangguran 5,3%. Tingkat Partisipasi Angkatan Kerja (TPAK) mencapai 63,7% dari penduduk usia kerja 8,52 juta jiwa. Sektor jasa mendominasi lapangan pekerjaan di Jakarta."
                },
                {
                    "id": "labor_002",
                    "province": "Jawa Barat", 
                    "description": "Jawa Barat memiliki angkatan kerja terbesar nasional 25,2 juta orang. Tingkat pengangguran 6,2% dengan TPAK 66,7%. Sektor manufaktur dan pertanian menjadi penyerap tenaga kerja utama. Penduduk usia kerja mencapai 37,8 juta jiwa."
                }
            ],
            "economic_data": [
                {
                    "id": "econ_001",
                    "province": "DKI Jakarta",
                    "description": "PDB DKI Jakarta 2024 mencapai Rp 3.250 triliun dengan PDB per kapita Rp 304 juta, tertinggi nasional. Inflasi terkendali 2,8% dengan tingkat kemiskinan rendah 3,5%. Koefisien Gini 0,42 menunjukkan kesenjangan menengah."
                },
                {
                    "id": "econ_002",
                    "province": "Jawa Barat",
                    "description": "PDB Jawa Barat Rp 2.180 triliun dengan per kapita Rp 44,5 juta. Kontributor terbesar ekonomi nasional. Inflasi 3,1%, kemiskinan 7,8%. Kesenjangan relatif rendah dengan Gini 0,38."
                }
            ]
        }
        
        return bps_data
    
    def create_text_chunks(self) -> tuple:
        """Create and return text chunks with metadata"""
        bps_data = self.load_bps_data()
        chunks = []
        metadata = []
        
        # Process each data category
        for category, items in bps_data.items():
            for item in items:
                chunk_text = f"Data {category.replace('_', ' ')}: {item['description']}"
                chunks.append(chunk_text)
                
                metadata.append({
                    "id": item["id"],
                    "province": item["province"],
                    "category": category,
                    "source": "BPS Indonesia"
                })
        
        return chunks, metadata

class VectorDatabase:
    """FAISS-based vector database"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.dimension = None
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        embeddings = self.embedding_model.encode(texts)
        return embeddings
    
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index"""
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> tuple:
        """Search for similar vectors"""
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        scores, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), k)
        return scores[0], indices[0]

class LLMGenerator:
    """LLM-based answer generation"""
    
    @staticmethod
    async def generate_answer(question: str, context_chunks: List[str]) -> str:
        """Generate answer using OpenAI GPT"""
        context = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""Berdasarkan data statistik BPS Indonesia berikut, jawab pertanyaan dengan akurat dan informatif.

Konteks dari BPS Indonesia:
{context}

Pertanyaan: {question}

Instruksi:
1. Jawab berdasarkan data yang disediakan
2. Jika data tidak tersedia, nyatakan dengan jelas
3. Sertakan angka dan statistik yang relevan
4. Jawab dalam bahasa Indonesia yang jelas dan profesional

Jawaban:"""

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Anda adalah asisten analisis data statistik BPS Indonesia yang memberikan jawaban akurat berdasarkan data yang tersedia."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# Initialize components
processor = DocumentProcessor()
vector_db = VectorDatabase(embedding_model)

@app.on_event("startup")
async def startup_event():
    """Initialize vector database on startup"""
    global faiss_index, text_chunks, chunk_metadata
    
    # Check if pre-built index exists
    if os.path.exists(VECTOR_DB_PATH) and os.path.exists(CHUNKS_PATH):
        print("Loading existing vector database...")
        with open(VECTOR_DB_PATH, 'rb') as f:
            vector_db.index = pickle.load(f)
        with open(CHUNKS_PATH, 'rb') as f:
            data = pickle.load(f)
            text_chunks = data['chunks']
            chunk_metadata = data['metadata']
    else:
        print("Building new vector database...")
        # Create text chunks
        text_chunks, chunk_metadata = processor.create_text_chunks()
        
        # Create embeddings
        embeddings = vector_db.create_embeddings(text_chunks)
        
        # Build FAISS index
        vector_db.build_index(embeddings)
        
        # Save to disk
        with open(VECTOR_DB_PATH, 'wb') as f:
            pickle.dump(vector_db.index, f)
        with open(CHUNKS_PATH, 'wb') as f:
            pickle.dump({'chunks': text_chunks, 'metadata': chunk_metadata}, f)
    
    print(f"Vector database ready with {len(text_chunks)} chunks")

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Main query endpoint for semantic similarity RAG"""
    try:
        # Create embedding for query
        query_embedding = vector_db.create_embeddings([request.question])
        
        # Search for similar chunks
        scores, indices = vector_db.search(query_embedding[0], k=request.top_k)
        
        # Get relevant chunks
        relevant_chunks = []
        context_texts = []
        
        for score, idx in zip(scores, indices):
            if idx < len(text_chunks):
                chunk_data = {
                    "text": text_chunks[idx],
                    "metadata": chunk_metadata[idx],
                    "similarity_score": float(score)
                }
                relevant_chunks.append(chunk_data)
                context_texts.append(text_chunks[idx])
        
        # Generate answer using LLM
        answer = await LLMGenerator.generate_answer(request.question, context_texts)
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            relevant_chunks=relevant_chunks,
            metadata={
                "model": "semantic_similarity",
                "embedding_model": EMBEDDING_MODEL,
                "chunks_found": len(relevant_chunks),
                "avg_similarity": float(np.mean(scores))
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def root():
    return {"message": "BPS Indonesia Semantic RAG System", "status": "active"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chunks_loaded": len(text_chunks),
        "vector_db_ready": vector_db.index is not None
    }

# Run with: uvicorn semantic_rag_system:app --reload --port 8000