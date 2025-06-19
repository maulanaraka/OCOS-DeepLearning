# semantic_rag.py
import os
import re
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

# Configure Gemini API - PENTING: Gunakan environment variable untuk API key
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBGfprRY9PW9PED_C_XcqPAZwHatxfMkbo")
genai.configure(api_key=API_KEY)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    relevant_chunks: List[str]

class SemanticRAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = None
        self.index = None
        # Coba model yang lebih baru
        try:
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            try:
                self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            except:
                # Fallback ke gemini-pro jika model lain tidak tersedia
                self.gemini_model = genai.GenerativeModel('gemini-pro')
        
    def load_and_process_data(self, file_path: str):
        """Load BPS data and split into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"Successfully loaded data from {file_path}")
        except FileNotFoundError:
            print(f"File {file_path} not found, using sample data")
            # Fallback sample data dengan data inflasi Indonesia yang lebih lengkap
            text = """
            Badan Pusat Statistik (BPS) adalah lembaga pemerintah yang bertanggung jawab dalam penyediaan data statistik.
            
            Inflasi Indonesia pada bulan Januari 2024 tercatat sebesar 2,57% year-on-year. Kenaikan inflasi ini terutama disebabkan oleh kenaikan harga bahan makanan dan energi. Inflasi inti tercatat 1,89% yoy.
            
            Pada bulan Februari 2024, inflasi Indonesia turun menjadi 2,75% year-on-year. Bank Indonesia memperkirakan inflasi akan tetap terkendali dalam target 3,0% ± 1%.
            
            Tingkat pengangguran terbuka di Indonesia pada Februari 2024 mencapai 5,32%. Angka ini menunjukkan penurunan dibandingkan periode yang sama tahun sebelumnya yang mencapai 5,45%.
            
            Produk Domestik Bruto (PDB) Indonesia tumbuh 5,04% pada kuartal IV 2023. Pertumbuhan ini didorong oleh konsumsi rumah tangga sebesar 4,91% dan investasi sebesar 4,2%.
            
            Ekspor Indonesia pada Desember 2023 mencapai USD 21,4 miliar. Komoditas utama ekspor meliputi kelapa sawit, batu bara, dan produk manufaktur. Total ekspor 2023 mencapai USD 291,9 miliar.
            
            Impor Indonesia pada periode yang sama tercatat USD 18,2 miliar. Barang impor utama adalah mesin, bahan kimia, dan produk elektronik. Total impor 2023 mencapai USD 238,8 miliar.
            
            Indeks Harga Konsumen (IHK) mengalami kenaikan 0,64% pada Januari 2024. Kenaikan tertinggi terjadi pada kelompok makanan dan minuman sebesar 1,2%.
            
            Nilai tukar rupiah terhadap dolar AS berada di level Rp 15.750 per USD pada akhir Januari 2024. Rupiah menguat 0,3% dibandingkan periode sebelumnya.
            
            Jumlah penduduk Indonesia diperkirakan mencapai 275,4 juta jiwa pada tahun 2024. Laju pertumbuhan penduduk sekitar 0,87% per tahun dengan kepadatan 146 jiwa per km².
            
            Tingkat kemiskinan di Indonesia turun menjadi 9,54% pada September 2023. Penurunan ini menunjukkan perbaikan kondisi ekonomi masyarakat dari 9,57% periode sebelumnya.
            
            Pada tahun 2024, di provinsi Aceh, terdapat 31740.0 pernikahan, 1192 kasus cerai talak, 4739 cerai gugat, dengan total 5931 perceraian.
            
            Pada tahun 2024, di provinsi Sumatera Utara, terdapat 66682.0 pernikahan, 2891 kasus cerai talak, 12861 cerai gugat, dengan total 15752 perceraian.
            
            Pada tahun 2024, di provinsi Sumatera Barat, terdapat 36486.0 pernikahan, 1706 kasus cerai talak, 6446 cerai gugat, dengan total 8152 perceraian.
            """
        except Exception as e:
            print(f"Error loading file: {e}")
            text = "Data tidak dapat dimuat."
        
        # Split into chunks (paragraphs)
        self.chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        print(f"Loaded {len(self.chunks)} chunks from data")
        
    def create_embeddings(self):
        """Create embeddings for all chunks"""
        print("Creating embeddings...")
        if not self.chunks:
            print("No chunks to process!")
            return
            
        self.embeddings = self.model.encode(self.chunks)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"Created FAISS index with {self.index.ntotal} vectors")
        
    def search_similar(self, query: str, top_k: int = 3) -> List[str]:
        """Search for similar chunks"""
        if self.index is None:
            return ["Data tidak tersedia"]
            
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        relevant_chunks = [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks)]
        return relevant_chunks
        
    def generate_answer(self, question: str, context: List[str]) -> str:
        """Generate answer using Gemini with better error handling"""
        context_text = "\n\n".join(context)
        
        prompt = f"""
        Berdasarkan informasi dari BPS berikut:
        
        {context_text}
        
        Pertanyaan: {question}
        
        Berikan jawaban yang akurat dan informatif berdasarkan data yang tersedia. Jika informasi tidak cukup, sebutkan bahwa data terbatas.
        Jawab dalam bahasa Indonesia dengan format yang jelas dan mudah dipahami.
        """
        
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=1000,
                    temperature=0.7,
                )
            )
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg and "not found" in error_msg:
                return "Maaf, terjadi error: Model AI sedang tidak tersedia. Coba lagi nanti atau hubungi administrator."
            elif "API" in error_msg:
                return "Maaf, terjadi error pada API. Periksa koneksi internet atau API key."
            else:
                return f"Maaf, terjadi error dalam generating jawaban: {error_msg}"
            
    def query(self, question: str) -> Dict:
        """Main query function"""
        if not self.chunks:
            return {
                "answer": "System belum siap. Data belum dimuat.",
                "relevant_chunks": []
            }
            
        relevant_chunks = self.search_similar(question)
        answer = self.generate_answer(question, relevant_chunks)
        
        return {
            "answer": answer,
            "relevant_chunks": relevant_chunks
        }

# Initialize RAG system
rag_system = SemanticRAG()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    print("Initializing BPS Semantic RAG...")
    rag_system.load_and_process_data("bps_faiss_texts.txt")  # Try to load your file
    rag_system.create_embeddings()
    print("RAG system ready!")
    yield
    # Shutdown
    print("Shutting down...")

# FastAPI app
app = FastAPI(
    title="BPS Semantic RAG API",
    description="API untuk mencari informasi statistik Indonesia dari BPS",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query endpoint for semantic RAG"""
    try:
        result = rag_system.query(request.question)
        return QueryResponse(
            answer=result["answer"],
            relevant_chunks=result["relevant_chunks"]
        )
    except Exception as e:
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "BPS Semantic RAG API is running",
        "status": "healthy",
        "endpoints": {
            "query": "/query (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chunks_loaded": len(rag_system.chunks),
        "embeddings_ready": rag_system.embeddings is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)