#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.dates import DateFormatter

# --------------------------------------


# __________________ New ______________________________________________________________________


df = pd.read_csv(
    'https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/01_Market_Making/df_main.csv',
    on_bad_lines='skip')

# Basic page setup
st.set_page_config(page_title="StableMM Radar", layout="wide")
st.title("📊 StableMM Radar — Market Making Insights on Stablecoin Pools")

# Sidebar filters
st.sidebar.header("🔍 Filters")
selected_chain = st.sidebar.multiselect("Select Chain(s)", options=df['blockchain'].unique(), default=list(df['blockchain'].unique()))
selected_dex = st.sidebar.multiselect("Select DEX(s)", options=df['dex'].unique(), default=list(df['dex'].unique()))
selected_pool = st.sidebar.multiselect("Select Pool(s)", options=df['pool'].unique(), default=list(df['pool'].unique()))
min_tvl = st.sidebar.slider("Minimum TVL (USD)", min_value=0, max_value=int(df['volume'].max()), value=50000000)
min_spread = st.sidebar.slider("Minimum Spread (%)", min_value=0.0, max_value=1.0, value=0.02)

# Filter Data
df_filtered = df[
    (df['blockchain'].isin(selected_chain)) &
    (df['dex'].isin(selected_dex)) &
    (df['pool'].isin(selected_pool)) &
    (df['volume'] > min_tvl) &
    (df['Spread'] > min_spread)
].copy()

# Calculate trade size
df_filtered['trade_size'] = df_filtered['volume'] / df_filtered['swap_count']

# --- Section 1: Slippage vs Trade Size Line Chart ---
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### 🧭 Slippage vs. Trade Size")
    st.markdown("""
    **Purpose**: Evaluate execution quality across different liquidity depths  
    **What to look for?** Watch for steep slopes — they signal thin liquidity and price impact risk.
    """)
with col2:
    fig, ax = plt.subplots(figsize=(10, 5))
    for pool_name, group in df_filtered.groupby("pool"):
        ax.plot(group["trade_size"], group["Spread"], label=pool_name)
    ax.set_xlabel("Trade Size (USD)")
    ax.set_ylabel("Spread (%)")
    ax.set_title("Slippage vs. Trade Size per Pool")
    ax.legend(fontsize="x-small")
    st.pyplot(fig)

# --- Section 2: Heatmap ---
col3, col4 = st.columns([1, 2])
with col3:
    st.markdown("### 🧊 Slippage Heatmap")
    st.markdown("""
    **Purpose**: Uncover execution weaknesses across pools  
    **What to look for?** Darker colors = higher slippage = liquidity gaps.
    """)
with col4:
    heatmap_data = df_filtered.pivot_table(
        index="pool", columns="order_size", values="Spread", aggfunc="mean"
    )
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax2)
    ax2.set_xlabel("Order Size (USD)")
    ax2.set_ylabel("Token Pair")
    ax2.set_title("Slippage Heatmap")
    st.pyplot(fig2)




# ـــ

st.markdown("---")
st.caption("Contact me: bellabahramii@gmail.com")


