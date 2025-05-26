import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load FAISS index and metadata
index = faiss.read_index("data/faiss_index.bin")
meta = pd.read_parquet("data/faiss_metadata.parquet")

# Prepare your query
model = SentenceTransformer('all-MiniLM-L6-v2')

# Edit this text for any search you want!
query_text = "best botox prices for anti-aging"
prompt = (
    f"Keyword: {query_text}\n"
    f"Group: Anti-Aging\n"
    f"Intent: transactional\n"
    f"Month: 2025-04\n"
    f"Clicks: 0, Impressions: 0, Ranking: 0.0, CTR: 0.0000, Clicks_MoM: 0.000, Impr_MoM: 0.000, Rank_MoM: 0.000"
)
query_vec = model.encode([prompt])
query_vec = query_vec / np.linalg.norm(query_vec)

# Run FAISS search
D, I = index.search(query_vec.astype('float32'), 5)

print("\n--- Top 5 Semantic Matches ---")
for idx, sim in zip(I[0], D[0]):
    print(f"Score: {sim:.4f}")
    print(meta.iloc[idx][["query", "keyword_group", "intent", "month", "ctr"]])
    print("-" * 50)
