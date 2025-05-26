import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Paths
ENRICHED_PATH = 'data/keywords_enriched.parquet'
OUTDIR = 'outputs/figures/'
os.makedirs(OUTDIR, exist_ok=True)

# 1. Load Data
df = pd.read_parquet(ENRICHED_PATH)

# Fix month to string for plotting
if 'month' in df.columns:
    df['month'] = df['month'].astype(str)
else:
    df['month'] = pd.to_datetime(df['month_start']).dt.to_period('M').astype(str)

print("\n==== Columns in Data ====")
print(df.columns)
print("\n==== Month/Query Sample ====")
print(df[['month_start', 'month', 'query']].head(10))

print("\n==== Data Sample Preview ====")
print(df[['query', 'month', 'clicks_mom', 'ctr', 'keyword_group', 'intent']].head(10))

# Filter out inf/-inf/nan
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['clicks_mom', 'ctr', 'month'])

# [1] Top Keywords by Clicks MoM (exclude nan/inf)
top_keywords = df.sort_values('clicks_mom', ascending=False).drop_duplicates('query').head(5)
print("\n==== [1] Top Keywords by Clicks MoM ====")
print(top_keywords[['query', 'month', 'clicks_mom']])

if not top_keywords.empty:
    plt.figure(figsize=(10, 6))
    for key in top_keywords['query']:
        df_kw = df[df['query'] == key].sort_values('month')
        # Convert month to datetime for correct x-axis
        x = pd.to_datetime(df_kw['month'], format='%Y-%m', errors='coerce')
        plt.plot(x, df_kw['clicks_mom'], marker='o', label=key)
    plt.xlabel('Month')
    plt.ylabel('Clicks MoM')
    plt.title('Top 5 Keywords: Month-over-Month Click Growth')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'top_keywords_trend.png'))
    plt.close()

# [2] Group Means by Clicks MoM
group_means = df.groupby('keyword_group')['clicks_mom'].mean().sort_values(ascending=False)
print("\n==== [2] Group Means by Clicks MoM ====")
print(group_means)
plt.figure(figsize=(10, 6))
sns.barplot(x=group_means.values, y=group_means.index, palette="viridis")
plt.xlabel('Average Clicks MoM')
plt.ylabel('Keyword Group')
plt.title('Average Clicks MoM by Keyword Group (Top 10)')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'group_avg_clicks_mom.png'))
plt.close()

# [3] CTR Trend for Weight Loss Group
wl_group = df[df['keyword_group'] == 'Weight Loss']
ctr_trend = wl_group.groupby('month')['ctr'].mean().reset_index()
print("\n==== [3] CTR Trend for Group: Weight Loss ====")
print(ctr_trend)
if len(ctr_trend) > 1 and ctr_trend['ctr'].notna().sum() > 0:
    plt.figure(figsize=(10, 6))
    x = pd.to_datetime(ctr_trend['month'], format='%Y-%m', errors='coerce')
    plt.plot(x, ctr_trend['ctr'], marker='o')
    plt.xlabel('Month')
    plt.ylabel('Average CTR')
    plt.title('Click-Through Rate Trend: Weight Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'ctr_trend_weight_loss.png'))
    plt.close()
else:
    print("[Warning] Not enough valid data for CTR trend plot of group 'Weight Loss'.")

# [4] Query Counts by Intent
intent_counts = df['intent'].value_counts()
print("\n==== [4] Query Counts by Intent ====")
print(intent_counts)
plt.figure(figsize=(8, 6))
sns.barplot(x=intent_counts.index, y=intent_counts.values, palette="deep")
plt.xlabel('Intent')
plt.ylabel('Count')
plt.title('Query Counts by Intent')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'intent_query_count.png'))
plt.close()

print("\nAll plots saved in outputs/figures/ (where possible).")
