import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

MODEL_PATH = "models/llama-3-8b-instruct.Q4_K_M.gguf"  # or your downloaded model
META_PATH = "data/faiss_metadata.parquet"

def get_llm():
    # Adjust n_ctx (context window) and n_gpu_layers (if you have a GPU)
    return Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, n_gpu_layers=20)

def generate_insight(llm, context, question):
    prompt = (
        "You are an expert SEO data analyst.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Give a detailed, concise, and actionable summary and recommendations.\n"
    )
    output = llm(prompt, max_tokens=512, stop=["</s>"])
    return output["choices"][0]["text"].strip()

def retrieve_context(query_text, top_k=8):
    # Load index and metadata
    index = faiss.read_index("data/faiss_index.bin")
    meta = pd.read_parquet(META_PATH)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    prompt = (
        f"Keyword: {query_text}\n"
        f"Group: \nIntent: \nMonth: 2025-04\nClicks: 0, Impressions: 0, Ranking: 0.0, CTR: 0.0000, "
        "Clicks_MoM: 0.000, Impr_MoM: 0.000, Rank_MoM: 0.000"
    )
    query_vec = model.encode([prompt])
    query_vec = query_vec / np.linalg.norm(query_vec)
    D, I = index.search(query_vec.astype('float32'), top_k)
    rows = []
    for idx in I[0]:
        row = meta.iloc[idx]
        rows.append(
            f"Query: {row['query']} | Group: {row['keyword_group']} | Intent: {row['intent']} | "
            f"Month: {row['month']} | Clicks_MoM: {row['clicks_mom']} | Rank_MoM: {row['rank_mom']} | CTR: {row['ctr']}"
        )
    return "\n".join(rows)

if __name__ == "__main__":
    print("Loading LLM... (first run may take 1-2 min)")
    llm = get_llm()
    print("LLM loaded!")

    # Example: Top improving keywords insight
    context = retrieve_context("keywords with most improvement in clicks in April 2025", top_k=8)
    question = "Which keywords improved the most and what does this suggest for our SEO?"
    print("\n---- INSIGHT: TOP IMPROVING KEYWORDS ----")
    print(generate_insight(llm, context, question))

    # Example: Keywords needing urgent attention
    context2 = retrieve_context("keywords with biggest drop in ranking or clicks", top_k=8)
    question2 = "Which keywords or groups need urgent SEO attention?"
    print("\n---- INSIGHT: DECLINING/NEEDS ATTENTION ----")
    print(generate_insight(llm, context2, question2))
