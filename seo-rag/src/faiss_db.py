import os
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm

INPUT_PATH = os.path.join('data', 'embeddings_for_pinecone.parquet')
INDEX_PATH = os.path.join('data', 'faiss_index.bin')
META_PATH = os.path.join('data', 'faiss_metadata.parquet')

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)

def build_faiss_index(df, embedding_col="embedding"):
    # Prepare matrix
    vectors = np.vstack(df[embedding_col].values).astype('float32')
    vectors = normalize(vectors)
    dim = vectors.shape[1]
    # Cosine similarity with Faiss = inner product on normalized vectors
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def save_index(index, path):
    faiss.write_index(index, path)

if __name__ == "__main__":
    # 1. Load data
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {df.shape[0]} vectors from {INPUT_PATH}")

    # 2. Build and save index
    index = build_faiss_index(df)
    save_index(index, INDEX_PATH)
    print(f"Saved FAISS index to {INDEX_PATH}")

    # 3. Save all metadata (excluding embeddings)
    meta_cols = [c for c in df.columns if c != "embedding"]
    df[meta_cols].to_parquet(META_PATH)
    print(f"Saved metadata to {META_PATH}")

    print("\nFAISS index and metadata are ready for local semantic search!")
