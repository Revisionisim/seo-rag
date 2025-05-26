import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "data/keywords_enriched.parquet"
st.set_page_config(layout="wide", page_title="SEO RAG Analytics")

st.title("SEO Analytics Dashboard")

# Load Data
@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)
df = load_data()

st.sidebar.header("Filter Data")
group = st.sidebar.selectbox("Keyword Group", ["All"] + sorted(df['keyword_group'].dropna().unique()))
intent = st.sidebar.selectbox("Intent", ["All"] + sorted(df['intent'].dropna().unique()))
month = st.sidebar.selectbox("Month", ["All"] + sorted(df['month'].astype(str).dropna().unique()))

filtered = df.copy()
if group != "All":
    filtered = filtered[filtered['keyword_group'] == group]
if intent != "All":
    filtered = filtered[filtered['intent'] == intent]
if month != "All":
    filtered = filtered[filtered['month'].astype(str) == month]

st.write(f"### Data Preview ({len(filtered)} rows)")
st.dataframe(filtered.head(20))

st.header("Visualizations")

# [1] Top 5 Keywords MoM Trend
st.subheader("Top 5 Keywords: Month-over-Month Click Growth")
top_keywords = filtered.groupby('query')['clicks_mom'].mean().sort_values(ascending=False).head(5).index.tolist()
df_top = filtered[filtered['query'].isin(top_keywords)].copy()
fig1, ax1 = plt.subplots(figsize=(8,5))
for key in top_keywords:
    df_kw = df_top[df_top['query'] == key].sort_values('month')
    ax1.plot(df_kw['month'].dt.strftime('%Y-%m'), df_kw['clicks_mom'], marker='o', label=key)
ax1.set_title("Top 5 Keywords: Month-over-Month Click Growth")
ax1.set_xlabel("Month")
ax1.set_ylabel("Clicks MoM")
ax1.legend()
st.pyplot(fig1)

# [2] Keyword Group: Avg Clicks MoM
st.subheader("Average Clicks MoM by Keyword Group")
group_means = filtered.groupby('keyword_group')['clicks_mom'].mean().sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.barplot(x=group_means.values, y=group_means.index, palette="viridis", ax=ax2)
ax2.set_title("Average Clicks MoM by Keyword Group")
ax2.set_xlabel("Average Clicks MoM")
ax2.set_ylabel("Keyword Group")
st.pyplot(fig2)

# [3] CTR Trend for Weight Loss
st.subheader("Click-Through Rate Trend: Weight Loss")
df_group = filtered[filtered['keyword_group'] == "Weight Loss"].copy()
if not df_group.empty:
    ctr_trend = df_group.groupby('month')['ctr'].mean().reset_index()
    fig3, ax3 = plt.subplots(figsize=(8,5))
    ax3.plot(ctr_trend['month'].dt.strftime('%Y-%m'), ctr_trend['ctr'], marker='o')
    ax3.set_title("CTR Trend: Weight Loss")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Average CTR")
    st.pyplot(fig3)
else:
    st.info("No data available for 'Weight Loss' group.")

# [4] Query Counts by Intent
st.subheader("Query Counts by Intent")
intent_counts = filtered['intent'].value_counts()
fig4, ax4 = plt.subplots(figsize=(8,5))
sns.barplot(x=intent_counts.index, y=intent_counts.values, palette="deep", ax=ax4)
ax4.set_title("Query Counts by Intent")
ax4.set_xlabel("Intent")
ax4.set_ylabel("Count")
st.pyplot(fig4)

st.success("All visualizations generated. Use sidebar to filter.")

