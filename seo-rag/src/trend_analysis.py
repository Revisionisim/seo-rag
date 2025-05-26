import pandas as pd

META_PATH = "data/faiss_metadata.parquet"
EMBED_PATH = "data/embeddings_for_pinecone.parquet"

def top_trending_keywords(df, group=None, intent=None, top_n=10, trend='improving'):
    """
    Returns top trending keywords by MoM metrics.
    - trend: 'improving' for positive clicks_mom, 'declining' for negative.
    - Optionally filter by group or intent.
    """
    df = df.copy()
    if group:
        df = df[df['keyword_group'] == group]
    if intent:
        df = df[df['intent'] == intent]
    df = df[df['clicks_mom'].notnull()]  # Remove rows where MoM is NaN

    # Choose sort order
    if trend == 'improving':
        df = df.sort_values("clicks_mom", ascending=False)
    else:
        df = df.sort_values("clicks_mom", ascending=True)
    return df.head(top_n)[['query', 'keyword_group', 'intent', 'month', 'clicks_mom', 'impr_mom', 'rank_mom', 'ctr']]

def trend_by_group(df, trend='improving', top_n=5):
    """
    Shows groups with highest average click/impression MoM.
    """
    group_trends = df.groupby('keyword_group')[['clicks_mom', 'impr_mom', 'rank_mom']].mean()
    if trend == 'improving':
        group_trends = group_trends.sort_values('clicks_mom', ascending=False)
    else:
        group_trends = group_trends.sort_values('clicks_mom', ascending=True)
    return group_trends.head(top_n)

def flexible_filter(df, **filters):
    """
    Supports custom pandas filters: group="Weight Loss", intent="transactional", month="2025-04", etc.
    """
    for col, val in filters.items():
        df = df[df[col] == val]
    return df

if __name__ == "__main__":
    # Load metadata (and optionally embeddings for search)
    df = pd.read_parquet(META_PATH)
    print(f"Loaded metadata: {df.shape}")

    print("\n=== TOP 10 IMPROVING KEYWORDS (by clicks MoM) ===")
    print(top_trending_keywords(df, trend='improving', top_n=10))

    print("\n=== TOP 10 DECLINING KEYWORDS (by clicks MoM) ===")
    print(top_trending_keywords(df, trend='declining', top_n=10))

    print("\n=== TOP 5 IMPROVING GROUPS (avg clicks MoM) ===")
    print(trend_by_group(df, trend='improving', top_n=5))

    print("\n=== TOP 5 DECLINING GROUPS (avg clicks MoM) ===")
    print(trend_by_group(df, trend='declining', top_n=5))

    print("\n=== TRENDING TRANSACTIONAL WEIGHT LOSS QUERIES ===")
    results = top_trending_keywords(df, group="Weight Loss", intent="transactional", trend='improving', top_n=5)
    print(results)

    print("\n=== FLEXIBLE FILTER EXAMPLE: informational, Facial & Laser Services, March 2025 ===")
    filtered = flexible_filter(df, intent="informational", keyword_group="Facial & Laser Services", month="2025-03")
    print(filtered[['query', 'clicks_mom', 'ctr', 'month']].head(10))

