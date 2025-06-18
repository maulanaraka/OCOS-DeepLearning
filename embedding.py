import pandas as pd
from sentence_transformers import SentenceTransformer

# Baca file CSV
df = pd.read_csv('Penduduk, Laju Pertumbuhan Penduduk, Distribusi Persentase Penduduk, Kepadatan Penduduk, dan Rasio Jenis Kelamin Penduduk Menurut Provinsi, 2020.csv')

# Gabungkan kolom yang ingin di-embed (misal: semua kolom jadi satu string per baris)
texts = df.astype(str).agg(' '.join, axis=1).tolist()

# Load model embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Proses embedding
embeddings = model.encode(texts)

# embeddings sekarang adalah array vektor untuk tiap baris
print(embeddings.shape)