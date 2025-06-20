# kg_rag.py
import re
import json
import os
from typing import Dict, List, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import networkx as nx
from contextlib import asynccontextmanager

# Configure Gemini API - PENTING: Gunakan environment variable untuk API key
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBGfprRY9PW9PED_C_XcqPAZwHatxfMkbo")
genai.configure(api_key=API_KEY)

class KGQueryRequest(BaseModel):
    question: str

class KGQueryResponse(BaseModel):
    answer: str
    explanation: str
    relevant_entities: List[str]
    relationships: List[Dict]

class KnowledgeGraphRAG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities = {}
        # Coba model yang lebih baru dengan fallback
        try:
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            print("Using gemini-1.5-flash model")
        except:
            try:
                self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
                print("Using gemini-1.5-pro model")
            except:
                # Fallback ke gemini-pro
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                print("Using gemini-pro model (fallback)")
        self.raw_data = ""
        
    def load_data(self, file_path: str):
        """Load BPS data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.raw_data = f.read()
            print(f"Successfully loaded data from {file_path}")
        except FileNotFoundError:
            print(f"File {file_path} not found, using enhanced sample data")
           # Enhanced sample data dengan lebih banyak informasi statistik Indonesia
            self.raw_data = """
            Inflasi Indonesia pada bulan Januari 2024 tercatat sebesar 2,57% year-on-year. Kenaikan inflasi ini terutama disebabkan oleh kenaikan harga bahan makanan dan energi.
             
            Inflasi inti Indonesia pada Januari 2024 tercatat sebesar 1,89% year-on-year, menunjukkan tekanan inflasi yang terkendali.
            
            Tingkat pengangguran terbuka di Indonesia pada Februari 2024 mencapai 5,32%. Angka ini menunjukkan penurunan dibandingkan periode yang sama tahun sebelumnya.
            
            Produk Domestik Bruto (PDB) Indonesia tumbuh 5,04% pada kuartal IV 2023. Pertumbuhan ini didorong oleh konsumsi rumah tangga dan investasi.
            
            Ekspor Indonesia pada Desember 2023 mencapai USD 21,4 miliar. Komoditas utama ekspor meliputi kelapa sawit, batu bara, dan produk manufaktur.
            
            Impor Indonesia pada periode yang sama tercatat USD 18,2 miliar. Barang impor utama adalah mesin, bahan kimia, dan produk elektronik.
            
            Indeks Harga Konsumen (IHK) mengalami kenaikan 0,64% pada Januari 2024. Kenaikan tertinggi terjadi pada kelompok makanan dan minuman.
            
            Nilai tukar rupiah terhadap dolar AS berada di level Rp 15.750 per USD pada akhir Januari 2024. Rupiah menguat 0,3% dibandingkan periode sebelumnya.
            
            Jumlah penduduk Indonesia diperkirakan mencapai 275,4 juta jiwa pada tahun 2024. Laju pertumbuhan penduduk sekitar 0,87% per tahun.
            
            Tingkat kemiskinan di Indonesia turun menjadi 9,54% pada September 2023. Penurunan ini menunjukkan perbaikan kondisi ekonomi masyarakat.
            
            Pada tahun 2024, di provinsi Aceh, terdapat 31740.0 pernikahan, 1192 kasus cerai talak, 4739 cerai gugat, dengan total 5931 perceraian.
            
            Pada tahun 2024, di provinsi Sumatera Utara, terdapat 66682.0 pernikahan, 2891 kasus cerai talak, 12861 cerai gugat, dengan total 15752 perceraian.
            
            Pada tahun 2024, di provinsi Sumatera Barat, terdapat 36486.0 pernikahan, 1706 kasus cerai talak, 6446 cerai gugat, dengan total 8152 perceraian.
            
            Bank Indonesia mempertahankan suku bunga acuan BI Rate sebesar 6,00% pada rapat bulan Januari 2024.
            
            Indeks Saham Gabungan (IHSG) ditutup pada level 7.267,55 pada akhir Januari 2024, menguat 1,2% dari periode sebelumnya.
            """
        except Exception as e:
            print(f"Error loading file: {e}")
            self.raw_data = "Data tidak dapat dimuat."
            
        print(f"Loaded data: {len(self.raw_data)} characters")
        
    def extract_entities_and_relationships(self):
        """Extract entities and build knowledge graph"""
        # Enhanced entities data dengan lebih banyak kategori
        entities_data = {
            # Countries and Regions
            "Indonesia": {"type": "Country", "description": "Negara dengan berbagai indikator ekonomi"},
            "Aceh": {"type": "Province", "description": "Provinsi di Indonesia"},
            "Sumatera_Utara": {"type": "Province", "description": "Provinsi di Indonesia"},
            "Sumatera_Barat": {"type": "Province", "description": "Provinsi di Indonesia"},
            
            # Economic Indicators
            "Inflasi": {"type": "Economic_Indicator", "description": "Tingkat kenaikan harga secara umum"},
            "Inflasi_Inti": {"type": "Economic_Indicator", "description": "Inflasi tanpa volatile food dan administered prices"},
            "Pengangguran": {"type": "Economic_Indicator", "description": "Tingkat pengangguran terbuka"},
            "PDB": {"type": "Economic_Indicator", "description": "Produk Domestik Bruto"},
            "Pertumbuhan_Ekonomi": {"type": "Economic_Indicator", "description": "Tingkat pertumbuhan ekonomi"},
            
            # Trade Indicators
            "Ekspor": {"type": "Trade_Indicator", "description": "Nilai ekspor barang dan jasa"},
            "Impor": {"type": "Trade_Indicator", "description": "Nilai impor barang dan jasa"},
            "Neraca_Perdagangan": {"type": "Trade_Indicator", "description": "Selisih ekspor dan impor"},
            
            # Price Indices
            "IHK": {"type": "Price_Index", "description": "Indeks Harga Konsumen"},
            "IHSG": {"type": "Stock_Index", "description": "Indeks Harga Saham Gabungan"},
            
            # Financial
            "Rupiah": {"type": "Currency", "description": "Mata uang Indonesia"},
            "BI_Rate": {"type": "Interest_Rate", "description": "Suku bunga acuan Bank Indonesia"},
            "Bank_Indonesia": {"type": "Institution", "description": "Bank sentral Indonesia"},
            
            # Demographics
            "Penduduk": {"type": "Demographic", "description": "Jumlah penduduk Indonesia"},
            "Pertumbuhan_Penduduk": {"type": "Demographic", "description": "Laju pertumbuhan penduduk"},
            
            # Social Indicators
            "Kemiskinan": {"type": "Social_Indicator", "description": "Tingkat kemiskinan"},
            "Pernikahan": {"type": "Social_Indicator", "description": "Jumlah pernikahan"},
            "Perceraian": {"type": "Social_Indicator", "description": "Jumlah perceraian"},
            
            # Commodities
            "Kelapa_Sawit": {"type": "Commodity", "description": "Komoditas ekspor utama Indonesia"},
            "Batu_Bara": {"type": "Commodity", "description": "Komoditas ekspor utama Indonesia"},
            "Manufaktur": {"type": "Sector", "description": "Sektor manufaktur"},
        }
        
        # Add entities to graph
        for entity, data in entities_data.items():
            self.graph.add_node(entity, **data)
            self.entities[entity] = data
            
        # Enhanced relationships dengan lebih banyak koneksi
        relationships = [
            # Indonesia's economic indicators
            ("Indonesia", "has_indicator", "Inflasi", {"value": "2.57%", "period": "Jan 2024", "trend": "yearly"}),
            ("Indonesia", "has_indicator", "Inflasi_Inti", {"value": "1.89%", "period": "Jan 2024", "trend": "controlled"}),
            ("Indonesia", "has_indicator", "Pengangguran", {"value": "5.32%", "period": "Feb 2024", "trend": "decreasing"}),
            ("Indonesia", "has_indicator", "PDB", {"value": "5.04%", "period": "Q4 2023", "trend": "growth"}),
            
            # Trade relationships
            ("Indonesia", "has_trade", "Ekspor", {"value": "21.4 billion USD", "period": "Dec 2023"}),
            ("Indonesia", "has_trade", "Impor", {"value": "18.2 billion USD", "period": "Dec 2023"}),
            ("Ekspor", "includes", "Kelapa_Sawit", {"type": "main_commodity"}),
            ("Ekspor", "includes", "Batu_Bara", {"type": "main_commodity"}),
            ("Ekspor", "includes", "Manufaktur", {"type": "main_commodity"}),
            
            # Financial relationships
            ("Indonesia", "has_index", "IHK", {"value": "0.64%", "period": "Jan 2024", "trend": "increasing"}),
            ("Indonesia", "has_currency", "Rupiah", {"rate": "15,750 per USD", "period": "Jan 2024"}),
            ("Bank_Indonesia", "sets", "BI_Rate", {"rate": "6.00%", "period": "Jan 2024", "status": "maintained"}),
            ("Indonesia", "has_stock_index", "IHSG", {"value": "7,267.55", "period": "Jan 2024", "change": "+1.2%"}),
            
            # Demographic relationships
            ("Indonesia", "has_population", "Penduduk", {"value": "275.4 million", "year": "2024", "growth": "0.87%"}),
            ("Indonesia", "has_social_indicator", "Kemiskinan", {"value": "9.54%", "period": "Sep 2023", "trend": "decreasing"}),
            
            # Provincial data
            ("Aceh", "belongs_to", "Indonesia", {"type": "province"}),
            ("Sumatera_Utara", "belongs_to", "Indonesia", {"type": "province"}),
            ("Sumatera_Barat", "belongs_to", "Indonesia", {"type": "province"}),
            ("Aceh", "has_marriages", "Pernikahan", {"value": "31,740", "year": "2024"}),
            ("Aceh", "has_divorces", "Perceraian", {"value": "5,931", "year": "2024"}),
            ("Sumatera_Utara", "has_marriages", "Pernikahan", {"value": "66,682", "year": "2024"}),
            ("Sumatera_Utara", "has_divorces", "Perceraian", {"value": "15,752", "year": "2024"}),
            ("Sumatera_Barat", "has_marriages", "Pernikahan", {"value": "36,486", "year": "2024"}),
            ("Sumatera_Barat", "has_divorces", "Perceraian", {"value": "8,152", "year": "2024"}),
            
            # Economic relationships
            ("Inflasi", "affects", "IHK", {"relationship": "directly_related"}),
            ("PDB", "indicates", "Pertumbuhan_Ekonomi", {"relationship": "positive_correlation"}),
            ("Ekspor", "contributes_to", "PDB", {"relationship": "positive_impact"}),
            ("Impor", "affects", "Neraca_Perdagangan", {"relationship": "negative_impact"}),
            ("BI_Rate", "influences", "Inflasi", {"relationship": "monetary_policy"}),
            ("Rupiah", "affects", "Ekspor", {"relationship": "exchange_rate_impact"}),
        ]
        
        # Add relationships to graph
        for source, relation, target, attributes in relationships:
            if target not in self.entities:
                self.graph.add_node(target, type="Concept", description=f"{target} concept")
                self.entities[target] = {"type": "Concept", "description": f"{target} concept"}
            
            self.graph.add_edge(source, target, relation=relation, **attributes)
            
        print(f"Knowledge Graph created with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        
    def find_relevant_entities(self, question: str) -> List[str]:
        """Find entities mentioned in the question with enhanced keyword matching"""
        question_lower = question.lower()
        relevant = []
        
        entity_keywords = {
            "inflasi": "Inflasi",
            "inflasi inti": "Inflasi_Inti",
            "pengangguran": "Pengangguran", 
            "pdb": "PDB",
            "produk domestik bruto": "PDB",
            "pertumbuhan ekonomi": "Pertumbuhan_Ekonomi",
            "ekspor": "Ekspor",
            "impor": "Impor",
            "neraca perdagangan": "Neraca_Perdagangan",
            "ihk": "IHK",
            "indeks harga konsumen": "IHK",
            "indeks harga": "IHK",
            "ihsg": "IHSG",
            "saham": "IHSG",
            "rupiah": "Rupiah",
            "nilai tukar": "Rupiah",
            "bi rate": "BI_Rate",
            "suku bunga": "BI_Rate",
            "bank indonesia": "Bank_Indonesia",
            "penduduk": "Penduduk",
            "populasi": "Penduduk",
            "kemiskinan": "Kemiskinan",
            "pernikahan": "Pernikahan",
            "nikah": "Pernikahan",
            "perceraian": "Perceraian", 
            "cerai": "Perceraian",
            "indonesia": "Indonesia",
            "aceh": "Aceh",
            "sumatera utara": "Sumatera_Utara",
            "sumut": "Sumatera_Utara",
            "sumatera barat": "Sumatera_Barat",
            "sumbar": "Sumatera_Barat",
            "kelapa sawit": "Kelapa_Sawit",
            "sawit": "Kelapa_Sawit",
            "batu bara": "Batu_Bara",
            "batubara": "Batu_Bara",
            "manufaktur": "Manufaktur"
        }
        
        for keyword, entity in entity_keywords.items():
            if keyword in question_lower:
                relevant.append(entity)
                
        return list(set(relevant)) if relevant else ["Indonesia"]
        
    def get_entity_relationships(self, entities: List[str]) -> List[Dict]:
        """Get relationships for relevant entities with better filtering"""
        relationships = []
        processed_pairs = set()
        
        for entity in entities:
            if entity in self.graph:
                # Get outgoing edges
                for target in self.graph.successors(entity):
                    pair = (entity, target)
                    if pair not in processed_pairs:
                        edge_data = self.graph[entity][target]
                        relationships.append({
                            "source": entity,
                            "target": target,
                            "relation": edge_data.get('relation', 'related_to'),
                            "attributes": {k: v for k, v in edge_data.items() if k != 'relation'}
                        })
                        processed_pairs.add(pair)
                    
                # Get incoming edges
                for source in self.graph.predecessors(entity):
                    pair = (source, entity)
                    if pair not in processed_pairs:
                        edge_data = self.graph[source][entity]
                        relationships.append({
                            "source": source,
                            "target": entity,
                            "relation": edge_data.get('relation', 'related_to'),
                            "attributes": {k: v for k, v in edge_data.items() if k != 'relation'}
                        })
                        processed_pairs.add(pair)
                    
        return relationships
        
    def generate_kg_answer(self, question: str, entities: List[str], relationships: List[Dict]) -> Tuple[str, str]:
        """Generate answer based on knowledge graph with better error handling"""
        # Prepare context from KG
        kg_context = f"Entities yang relevan: {', '.join(entities)}\n\n"
        kg_context += "Relationships dari Knowledge Graph:\n"
        
        for rel in relationships:
            kg_context += f"- {rel['source']} --[{rel['relation']}]--> {rel['target']}"
            if rel['attributes']:
                attrs_str = ", ".join([f"{k}: {v}" for k, v in rel['attributes'].items()])
                kg_context += f" ({attrs_str})"
            kg_context += "\n"
            
        prompt = f"""
        Berdasarkan Knowledge Graph BPS berikut:
        
        {kg_context}
        
        Data mentah pendukung:
        {self.raw_data}
        
        Pertanyaan: {question}
        
        Berikan dua bagian jawaban:
        1. JAWABAN: Jawaban langsung untuk pertanyaan berdasarkan data yang ada
        2. PENJELASAN: Jelaskan bagaimana Knowledge Graph membantu menjawab pertanyaan ini, sebutkan entities dan relationships yang digunakan
        
        Format:
        JAWABAN: [jawaban singkat dan akurat]
        
        PENJELASAN: [penjelasan detail tentang reasoning dari KG]
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
            full_response = response.text
            
            # Split answer and explanation
            parts = full_response.split("PENJELASAN:")
            if len(parts) == 2:
                answer = parts[0].replace("JAWABAN:", "").strip()
                explanation = parts[1].strip()
            else:
                # Fallback if format not followed
                lines = full_response.split('\n')
                answer = lines[0] if lines else "Informasi tidak tersedia"
                explanation = "Knowledge Graph menghubungkan entities dan relationships yang relevan untuk memberikan jawaban komprehensif."
                
            return answer, explanation
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg and "not found" in error_msg:
                return "Maaf, terjadi error: Model AI sedang tidak tersedia. Coba lagi nanti.", "Error pada model AI"
            elif "API" in error_msg:
                return "Maaf, terjadi error pada API. Periksa koneksi internet atau API key.", "Error API"
            else:
                return f"Error generating answer: {error_msg}", "Terjadi kesalahan dalam pemrosesan Knowledge Graph"
            
    def query(self, question: str) -> Dict:
        """Main query function for KG RAG"""
        if not self.entities:
            return {
                "answer": "Knowledge Graph belum siap. System belum diinisialisasi.",
                "explanation": "Entities dan relationships belum dimuat.",
                "relevant_entities": [],
                "relationships": []
            }
            
        entities = self.find_relevant_entities(question)
        relationships = self.get_entity_relationships(entities)
        answer, explanation = self.generate_kg_answer(question, entities, relationships)
        
        return {
            "answer": answer,
            "explanation": explanation,
            "relevant_entities": entities,
            "relationships": relationships[:10]  # Limit to top 10 relationships
        }

# Initialize KG RAG system
kg_rag_system = KnowledgeGraphRAG()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    print("Initializing BPS Knowledge Graph RAG...")
    kg_rag_system.load_data("bps_faiss_texts.txt")
    kg_rag_system.extract_entities_and_relationships()
    print("Knowledge Graph RAG system ready!")
    yield
    # Shutdown
    print("Shutting down KG RAG system...")

# FastAPI app
app = FastAPI(
    title="BPS Knowledge Graph RAG API",
    description="API untuk analisis data statistik Indonesia menggunakan Knowledge Graph",
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

@app.post("/query-kg", response_model=KGQueryResponse)
async def query_kg_endpoint(request: KGQueryRequest):
    """Query endpoint for Knowledge Graph RAG"""
    try:
        result = kg_rag_system.query(request.question)
        return KGQueryResponse(
            answer=result["answer"],
            explanation=result["explanation"],
            relevant_entities=result["relevant_entities"],
            relationships=result["relationships"]
        )
    except Exception as e:
        print(f"KG Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "BPS Knowledge Graph RAG API is running",
        "status": "healthy",
        "endpoints": {
            "query-kg": "/query-kg (POST)",
            "graph-stats": "/graph-stats (GET)",
            "docs": "/docs"
        }
    }

@app.get("/graph-stats")
async def graph_stats():
    """Get knowledge graph statistics"""
    try:
        entity_types = {}
        for entity, data in kg_rag_system.entities.items():
            entity_type = data.get('type', 'Unknown')
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity)
            
        return {
            "nodes": len(kg_rag_system.graph.nodes),
            "edges": len(kg_rag_system.graph.edges),
            "entity_types": entity_types,
            "sample_entities": list(kg_rag_system.entities.keys())[:10]
        }
    except Exception as e:
        return {"error": f"Unable to get graph stats: {str(e)}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "entities_count": len(kg_rag_system.entities),
        "graph_ready": len(kg_rag_system.graph.nodes) > 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)