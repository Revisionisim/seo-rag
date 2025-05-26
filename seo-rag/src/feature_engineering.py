import pandas as pd
import os
import json

# ==== Utilities for intent & group mapping ====
def load_patterns(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def classify_intent(query, patterns):
    """Return highest-matching intent type for a query, or 'unknown'."""
    query_l = str(query).lower()
    for intent, keywords in patterns.items():
        for k in keywords:
            if k in query_l:
                return intent
    return "unknown"

def map_group(query, groups):
    """Return group name if any group keyword is present in query, else 'Misc Treatments'."""
    query_l = str(query).lower()
    for group, keywords in groups.items():
        for k in keywords:
            if k in query_l:
                return group
    return "Misc Treatments"

# ==== Main Feature Engineering Pipeline ====
def feature_engineer(df):
    # Compute CTR
    df['ctr'] = df['total_clicks'] / df['total_impressions']
    df['ctr'] = df['ctr'].replace([float('inf'), -float('inf')], 0).fillna(0)

    # Extract month for time-based grouping
    df['month'] = pd.to_datetime(df['month_start']).dt.to_period('M')
    df = df.sort_values(['query', 'month'])

    # Month-over-month changes
    df['clicks_mom'] = df.groupby('query')['total_clicks'].pct_change().fillna(0)
    df['impr_mom']   = df.groupby('query')['total_impressions'].pct_change().fillna(0)
    # For ranking: Lower value = better rank; so invert delta for "improvement"
    df['rank_mom']   = df.groupby('query')['avg_rankning'].diff().fillna(0) * -1

    return df

def add_intent_and_group(df, intent_patterns, group_patterns):
    df['intent'] = df['query'].apply(lambda x: classify_intent(x, intent_patterns))
    df['keyword_group'] = df['query'].apply(lambda x: map_group(x, group_patterns))
    return df

if __name__ == '__main__':
    # File paths
    RAW_PATH    = os.path.join('data', 'keywords_processed.csv')
    OUT_PATH    = os.path.join('data', 'keywords_enriched.csv')
    INTENT_PATH = os.path.join('data', 'dictionaries', 'intent_patterns.json')
    GROUPS_PATH = os.path.join('data', 'dictionaries', 'keyword_groups.json')

    # 1. Load cleaned data
    df = pd.read_parquet(RAW_PATH)
    print("Loaded cleaned data:", df.shape)

    # 2. Feature engineering
    df = feature_engineer(df)
    print("Feature engineering done.")

    # 3. Load intent and group dictionaries
    intent_patterns = load_patterns(INTENT_PATH)
    group_patterns  = load_patterns(GROUPS_PATH)

    # 4. Add intent and group columns
    df = add_intent_and_group(df, intent_patterns, group_patterns)
    print("Intent and group mapping done.")

    # 5. Save output
    df.to_parquet(OUT_PATH, index=False)
    print(f"Enriched data saved to {OUT_PATH}\n")
    print(df.head())
