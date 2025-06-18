import pandas as pd
from sentence_transformers import SentenceTransformer
import re
import faiss
import numpy as np
import glob

# Preprocessing function for Indonesian text
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

# List all CSV files
csv_files = glob.glob('*.csv')

all_texts = []
for file in csv_files:
    df = pd.read_csv(file)
    texts = df.astype(str).agg(' '.join, axis=1).tolist()
    texts = [preprocess_text(t) for t in texts]
    all_texts.extend(texts)

# Load Indonesian embedding model (or fallback to multilingual if not available)
try:
    model = SentenceTransformer('indobenchmark/indobert-base-p1')
except Exception:
    print('Indonesian model not found, using multilingual model instead.')
    model = SentenceTransformer('all-MiniLM-L6-v2')

# Proses embedding untuk semua teks dari semua file
embeddings = model.encode(all_texts)

# Store embeddings in FAISS
# Pastikan embeddings dalam format float32
embeddings = np.array(embeddings).astype('float32')

# Membuat index FAISS (menggunakan L2/Euclidean distance)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Simpan index ke file
faiss.write_index(index, 'faiss_index.idx')
print(f'FAISS index saved to faiss_index.idx with {embeddings.shape[0]} vectors from {len(csv_files)} files')
