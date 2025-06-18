# knowledge_graph_rag.py
# Ontology/Knowledge Graph-based RAG QA System for BPS Indonesia Data

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import networkx as nx
import json
import spacy
import openai
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict

# Initialize FastAPI app
app = FastAPI(title="BPS Indonesia Knowledge Graph RAG System", version="1.0.0")

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
openai.api_key = OPENAI_API_KEY

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

class QueryRequest(BaseModel):
    question: str
    max_depth: int = 2

class QueryResponse(BaseModel):
    question: str
    answer: str
    reasoning_path: List[Dict[str, Any]]
    relevant_entities: List[str]
    metadata: Dict[str, Any]

class BPSOntology:
    """Define BPS Indonesia data ontology"""
    
    def __init__(self):
        self.entity_types = {
            'Province': ['DKI Jakarta', 'Jawa Barat', 'Jawa Timur', 'Jawa Tengah', 'Sumatera Utara'],
            'Economic_Indicator': ['GDP', 'GDP_per_capita', 'inflation_rate', 'poverty_rate', 'gini_coefficient'],
            'Population_Indicator': ['population', 'growth_rate', 'density', 'male_percentage', 'female_percentage'],
            'Labor_Indicator': ['labor_force', 'employment_rate', 'unemployment_rate', 'participation_rate'],
            'Education_Indicator': ['literacy_rate', 'school_enrollment', 'university_graduates'],
            'Year': ['2023', '2024'],
            'Sector': ['manufacturing', 'agriculture', 'services', 'trade']
        }
        
        self.relationships = {
            'has_population': ('Province', 'Population_Indicator'),
            'has_economy': ('Province', 'Economic_Indicator'),
            'has_labor': ('Province', 'Labor_Indicator'),
            'has_education': ('Province', 'Education_Indicator'),
            'measured_in': ('*', 'Year'),
            'compares_to': ('Province', 'Province'),
            'higher_than': ('*', '*'),
            'lower_than': ('*', '*'),
            'part_of': ('*', 'Sector')
        }

class KnowledgeGraph:
    """Knowledge Graph implementation using NetworkX"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.ontology = BPSOntology()
        self.entity_data = {}
        
    def build_graph_from_bps_data(self):
        """Build knowledge graph from BPS data"""
        bps_data = self._load_bps_data()
        
        # Add provinces as nodes
        provinces = ['DKI Jakarta', 'Jawa Barat', 'Jawa Timur', 'Jawa Tengah', 'Sumatera Utara']
        for province in provinces:
            self.graph.add_node(province, type='Province')
        
        # Process population data
        for item in bps_data.get('population_data', []):
            province = item['province']
            
            # Add population indicators
            pop_node = f"{province}_population_2024"
            self.graph.add_node(pop_node, type='Population_Data', value=item['population_2024'])
            self.graph.add_edge(province, pop_node, relation='has_population')
            
            # Add growth rate
            growth_node = f"{province}_growth_rate"
            self.graph.add_node(growth_node, type='Growth_Rate', value=item['growth_rate'])
            self.graph.add_edge(province, growth_node, relation='has_growth_rate')
            
            # Add density
            density_node = f"{province}_density"
            self.graph.add_node(density_node, type='Population_Density', value=item['density_per_km2'])
            self.graph.add_edge(province, density_node, relation='has_density')
        
        # Process labor data
        for item in bps_data.get('labor_data', []):
            province = item['province']
            
            # Add unemployment rate
            unemployment_node = f"{province}_unemployment"
            self.graph.add_node(unemployment_node, type='Unemployment_Rate', value=item['unemployment_rate'])
            self.graph.add_edge(province, unemployment_node, relation='has_unemployment')
            
            # Add labor force
            labor_node = f"{province}_labor_force"
            self.graph.add_node(labor_node, type='Labor_Force', value=item['labor_force'])
            self.graph.add_edge(province, labor_node, relation='has_labor_force')
        
        # Process economic data
        for item in bps_data.get('economic_data', []):
            province = item['province']
            
            # Add GDP
            gdp_node = f"{province}_gdp"
            self.graph.add_node(gdp_node, type='GDP', value=item['gdp_2024'])
            self.graph.add_edge(province, gdp_node, relation='has_gdp')
            
            # Add GDP per capita
            gdp_per_capita_node = f"{province}_gdp_per_capita"
            self.graph.add_node(gdp_per_capita_node, type='GDP_Per_Capita', value=item['gdp_per_capita'])
            self.graph.add_edge(province, gdp_per_capita_node, relation='has_gdp_per_capita')
            
            # Add poverty rate
            poverty_node = f"{province}_poverty"
            self.graph.add_node(poverty_node, type='Poverty_Rate', value=item['poverty_rate'])
            self.graph.add_edge(province, poverty_node, relation='has_poverty_rate')
        
        # Add comparative relationships
        self._add_comparative_relationships()
        
        print(f"Knowledge graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def _load_bps_data(self):
        """Load BPS data"""
        return {
            "population_data": [
                {"province": "DKI Jakarta", "population_2024": 10679000, "growth_rate": 1.11, "density_per_km2": 16030},
                {"province": "Jawa Barat", "population_2024": 48939000, "growth_rate": 0.53, "density_per_km2": 1386},
                {"province": "Jawa Timur", "population_2024": 39871000, "growth_rate": 0.35, "density_per_km2": 840},
                {"province": "Jawa Tengah", "population_2024": 36617000, "growth_rate": 0.28, "density_per_km2": 1127},
                {"province": "Sumatera Utara", "population_2024": 15434000, "growth_rate": 0.88, "density_per_km2": 214}
            ],
            "labor_data": [
                {"province": "DKI Jakarta", "labor_force": 5430000, "unemployment_rate": 5.3},
                {"province": "Jawa Barat", "labor_force": 25200000, "unemployment_rate": 6.2},
                {"province": "Jawa Timur", "labor_force": 20850000, "unemployment_rate": 4.9},
                {"province": "Jawa Tengah", "labor_force": 18900000, "unemployment_rate": 4.6},
                {"province": "Sumatera Utara", "labor_force": 7850000, "unemployment_rate": 5.8}
            ],
            "economic_data": [
                {"province": "DKI Jakarta", "gdp_2024": 3250000000, "gdp_per_capita": 304000000, "poverty_rate": 3.5},
                {"province": "Jawa Barat", "gdp_2024": 2180000000, "gdp_per_capita": 44500000, "poverty_rate": 7.8},
                {"province": "Jawa Timur", "gdp_2024": 1950000000, "gdp_per_capita": 48900000, "poverty_rate": 10.2},
                {"province": "Jawa Tengah", "gdp_2024": 1420000000, "gdp_per_capita": 38800000, "poverty_rate": 11.5},
                {"province": "Sumatera Utara", "gdp_2024": 785000000, "gdp_per_capita": 50900000, "poverty_rate": 8.9}
            ]
        }
    
    def _add_comparative_relationships(self):
        """Add comparative relationships between provinces"""
        provinces = ['DKI Jakarta', 'Jawa Barat', 'Jawa Timur', 'Jawa Tengah', 'Sumatera Utara']
        
        # Add population comparisons
        pop_data = [
            ("DKI Jakarta", 10679000),
            ("Jawa Barat", 48939000),
            ("Jawa Timur", 39871000),
            ("Jawa Tengah", 36617000),
            ("Sumatera Utara", 15434000)
        ]
        
        # Sort by population
        pop_data.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(len(pop_data) - 1):
            higher_pop = pop_data[i][0]
            lower_pop = pop_data[i + 1][0]
            self.graph.add_edge(higher_pop, lower_pop, relation='higher_population_than')
    
    def extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entities from user query"""
        entities = []
        query_lower = query.lower()
        
        # Check for provinces
        for province in self.ontology.entity_types['Province']:
            if province.lower() in query_lower:
                entities.append(province)
        
        # Check for indicators
        indicator_keywords = {
            'population': 'Population_Data',
            'penduduk': 'Population_Data',
            'unemployment': 'Unemployment_Rate',
            'pengangguran': 'Unemployment_Rate',
            'gdp': 'GDP',
            'pdb': 'GDP',
            'poverty': 'Poverty_Rate',
            'kemiskinan': 'Poverty_Rate',
            'labor': 'Labor_Force',
            'tenaga kerja': 'Labor_Force',
            'density': 'Population_Density',
            'kepadatan': 'Population_Density'
        }
        
        for keyword, entity_type in indicator_keywords.items():
            if keyword in query_lower:
                entities.append(entity_type)
        
        return entities
    
    def query_subgraph(self, entities: List[str], max_depth: int = 2) -> Tuple[nx.MultiDiGraph, List[Dict]]:
        """Query relevant subgraph based on entities"""
        subgraph = nx.MultiDiGraph()
        reasoning_path = []
        
        # Find all relevant nodes
        relevant_nodes = set()
        for entity in entities:
            if entity in self.graph.nodes():
                relevant_nodes.add(entity)
                # Add neighbors
                for neighbor in self.graph.neighbors(entity):
                    relevant_nodes.add(neighbor)
                    
                    # Add edge information to reasoning path
                    for edge_data in self.graph[entity][neighbor].values():
                        reasoning_path.append({
                            'from': entity,
                            'to': neighbor,
                            'relation': edge_data.get('relation', 'related_to'),
                            'type': 'direct_connection'
                        })
        
        # Build subgraph
        subgraph.add_nodes_from(relevant_nodes)
        for node in relevant_nodes:
            if node in self.graph:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in relevant_nodes:
                        # Copy all edges between relevant nodes
                        for key, edge_data in self.graph[node][neighbor].items():
                            subgraph.add_edge(node, neighbor, key=key, **edge_data)
        
        return subgraph, reasoning_path
    
    def get_node_data(self, node: str) -> Dict:
        """Get data for a specific node"""
        if node in self.graph.nodes():
            node_data = self.graph.nodes[node].copy()
            node_data['node_id'] = node
            return node_data
        return {}

class QueryProcessor:
    """Process queries and generate responses"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
    
    async def process_query(self, question: str, max_depth: int = 2) -> Dict[str, Any]:
        """Process query and return structured response"""
        # Extract entities from question
        entities = self.kg.extract_entities_from_query(question)
        
        # Query knowledge graph
        subgraph, reasoning_path = self.kg.query_subgraph(entities, max_depth)
        
        # Collect relevant data for context
        context_data = []
        for node in subgraph.nodes():
            node_data = self.kg.get_node_data(node)
            if node_data:
                context_data.append(node_data)
        
        # Generate answer using LLM
        answer = await self._generate_answer_with_kg_context(question, context_data, reasoning_path)
        
        return {
            'question': question,
            'answer': answer,
            'reasoning_path': reasoning_path,
            'relevant_entities': entities,
            'context_data': context_data
        }
    
    async def _generate_answer_with_kg_context(self, question: str, context_data: List[Dict], reasoning_path: List[Dict]) -> str:
        """Generate answer using knowledge graph context"""
        
        # Format context from knowledge graph
        context_text = "Data dari Knowledge Graph BPS Indonesia:\n\n"
        
        for item in context_data:
            if 'value' in item and 'type' in item:
                context_text += f"- {item['node_id']}: {item['type']} = {item['value']}\n"
        
        # Format reasoning path
        reasoning_text = "\nJalur reasoning:\n"
        for step in reasoning_path[:5]:  # Limit to 5 steps
            reasoning_text += f"- {step['from']} --[{step['relation']}]--> {step['to']}\n"
        
        prompt = f"""Berdasarkan Knowledge Graph BPS Indonesia berikut, jawab pertanyaan dengan akurat.

{context_text}

{reasoning_text}

Pertanyaan: {question}

Instruksi:
1. Gunakan data dari knowledge graph yang disediakan
2. Jelaskan reasoning berdasarkan relasi antar entitas
3. Sertakan angka spesifik yang relevan
4. Jika perlu perbandingan, gunakan data yang tersedia
5. Jawab dalam bahasa Indonesia yang profesional

Jawaban:"""

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Anda adalah sistem analisis knowledge graph BPS Indonesia yang memberikan jawaban berdasarkan entitas dan relasi dalam graph."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# Initialize components
kg = KnowledgeGraph()
query_processor = QueryProcessor(kg)

@app.on_event("startup")
async def startup_event():
    """Initialize knowledge graph on startup"""
    print("Building knowledge graph...")
    kg.build_graph_from_bps_data()
    print("Knowledge graph ready!")

@app.post("/query-kg", response_model=QueryResponse)
async def query_kg_endpoint(request: QueryRequest):
    """Main query endpoint for knowledge graph RAG"""
    try:
        result = await query_processor.process_query(request.question, request.max_depth)
        
        return QueryResponse(
            question=result['question'],
            answer=result['answer'],
            reasoning_path=result['reasoning_path'],
            relevant_entities=result['relevant_entities'],
            metadata={
                "model": "knowledge_graph",
                "entities_found": len(result['relevant_entities']),
                "reasoning_steps": len(result['reasoning_path']),
                "graph_nodes": kg.graph.number_of_nodes(),
                "graph_edges": kg.graph.number_of_edges()
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def root():
    return {"message": "BPS Indonesia Knowledge Graph RAG System", "status": "active"}

@app.get("/graph-info")
async def graph_info():
    """Get knowledge graph information"""
    return {
        "nodes": kg.graph.number_of_nodes(),
        "edges": kg.graph.number_of_edges(),
        "node_types": list(kg.ontology.entity_types.keys()),
        "sample_nodes": list(kg.graph.nodes())[:10]
    }

# Run with: uvicorn knowledge_graph_rag:app --reload --port 8001