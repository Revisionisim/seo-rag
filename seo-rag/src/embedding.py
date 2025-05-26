import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

# Model - this is downloaded automatically if not present
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast and strong for general tasks

INPUT_PATH = os.path.join('data', 'keywords_enriched.parquet')
OUTPUT_PATH = os.path.join('data', 'embeddings_for_pinecone.parquet')

def build_prompt(row):
    # Merge all info as descriptive text for embedding 
    desc = (
        f"Keyword: {row['query']}\n"
        f"Group: {row['keyword_group']}\n"
        f"Intent: {row['intent']}\n"
        f"Month: {row['month']}\n"
        f"Clicks: {row['total_clicks']}, "
        f"Impressions: {row['total_impressions']}, "
        f"Ranking: {row['avg_rankning']:.2f}, "
        f"CTR: {row['ctr']:.4f}, "
        f"Clicks_MoM: {row['clicks_mom']:.3f}, "
        f"Impr_MoM: {row['impr_mom']:.3f}, "
        f"Rank_MoM: {row['rank_mom']:.3f}"
    )
    return desc

def batch_embeddings(df, model, batch_size=128):
    # Use model.encode for efficient batching
    prompts = df.apply(build_prompt, axis=1).tolist()
    embeddings = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating embeddings"):
        batch_prompts = prompts[i:i+batch_size]
        batch_emb = model.encode(batch_prompts, show_progress_bar=False)
        embeddings.extend(batch_emb)
    return np.array(embeddings)

if __name__ == "__main__":
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {df.shape[0]} rows from {INPUT_PATH}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = batch_embeddings(df, model)
    df['embedding'] = embeddings.tolist()
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved embeddings and metadata to {OUTPUT_PATH}")
    print(df[['query', 'keyword_group', 'intent', 'embedding']].head())
