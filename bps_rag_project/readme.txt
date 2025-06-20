# ===========================================
# SETUP INSTRUCTIONS
# ===========================================

## 1. Install Dependencies
pip install -r requirements.txt

## 2. Prepare Data File
# Pastikan file 'bps.txt' ada di direktori yang sama dengan script Python

## 3. Jalankan Semantic RAG (Part 2.1)
# Terminal 1:
python semantic_rag.py
# Server akan berjalan di http://localhost:8000
# Buka semantic_ui.html di browser

## 4. Jalankan Knowledge Graph RAG (Part 2.2)  
# Terminal 2:
python kg_rag.py
# Server akan berjalan di http://localhost:8001
# Buka kg_ui.html di browser

## 5. Testing APIs
# Semantic RAG:
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Berapa tingkat inflasi Indonesia?"}'

# Knowledge Graph RAG:
curl -X POST "http://localhost:8001/query-kg" \
     -H "Content-Type: application/json" \
     -d '{"question": "Bagaimana hubungan inflasi dengan IHK?"}'

## 6. Features Implemented

### Part 2.1 - Semantic Similarity RAG:
✅ Data scraping/loading dari file BPS.txt
✅ Text preprocessing dan chunking
✅ Vector embeddings menggunakan Sentence Transformers
✅ FAISS vector database untuk similarity search
✅ FastAPI backend dengan endpoint POST /query
✅ UI sederhana untuk Q&A dengan hasil embedding search
✅ Integration dengan Gemini API untuk answer generation

### Part 2.2 - Knowledge Graph RAG:
✅ Ontology/Knowledge Graph berbasis entities BPS
✅ Extraction entities dan relationships dari data
✅ NetworkX untuk graph operations
✅ FastAPI backend dengan endpoint POST /query-kg
✅ UI yang menampilkan:
   - Answer dari KG reasoning
   - Explanation bagaimana KG membantu menjawab
   - Relevant entities yang digunakan
   - Relationships dalam graph
✅ Graph statistics endpoint

## 7. Architecture Overview

### Semantic RAG Flow:
Input Question → Embedding → FAISS Search → Retrieve Chunks → Gemini Generate Answer

### Knowledge Graph RAG Flow:
Input Question → Entity Extraction → Graph Traversal → Relationship Analysis → Gemini Generate Answer + Explanation

## 8. File Structure
semantic_rag.py     # Part 2.1 implementation
semantic_ui.html    # UI for semantic RAG
kg_rag.py          # Part 2.2 implementation  
kg_ui.html         # UI for KG RAG
requirements.txt   # Dependencies
bps.txt           # Your BPS data file

## 9. API Endpoints

### Semantic RAG (Port 8000):
- GET /                     # Health check
- POST /query               # Main query endpoint

### Knowledge Graph RAG (Port 8001):
- GET /                     # Health check  
- POST /query-kg            # KG query endpoint
- GET /graph-stats          # Graph statistics

## 10. Sample Questions to Test:

- "Pertumbuhan penduduk Indonesia di tahun 2024"
- "Perbandingan jumlah penduduk jawa timur dan jawa barat terhadap seluruh indonesia per tahun"
- "Persentase perceraian di DKI Jakarta terhadap perceraian di seluruh indonesia per tahun"