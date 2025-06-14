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


data = pd.read_csv('https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/01_Market_Making/df_main.csv', on_bad_lines='skip')

# Add computed columns
data['trade_size'] = data['volume'] / data['swap_count']
data['trade_size_bin'] = pd.cut(data['trade_size'], bins=[0, 1e4, 5e4, 1e5, 5e5, 1e6, np.inf], 
                                labels=["<10k", "10k-50k", "50k-100k", "100k-500k", "500k-1M", "1M+"])
data['order_size_bin'] = pd.cut(data['order_size'], bins=[0, 1e4, 5e4, 1e5, 5e5, 1e6, np.inf],
                                labels=["<10k", "10k-50k", "50k-100k", "100k-500k", "500k-1M", "1M+"])

# Set up dashboard
st.set_page_config(page_title="StableMM Radar", layout="wide")
st.title("📊 StableMM Radar — Market Making Insights on Stablecoin Pools")

# Filters (now at the top)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    chain = st.selectbox("Select Chain", sorted(data['blockchain'].unique()))
with col2:
    dex = st.selectbox("Select DEX", sorted(data['dex'].unique()))
with col3:
    pool = st.selectbox("Select Pool", sorted(data['pool'].unique()))
with col4:
    min_tvl = st.number_input("Minimum TVL", min_value=0, value=50_000_000, step=1_000_000)
with col5:
    min_spread = st.slider("Min Spread (%)", 0.0, 1.0, 0.02)

# Apply filters
filtered = data[
    (data['blockchain'] == chain) &
    (data['dex'] == dex) &
    (data['pool'] == pool) &
    (data['volume'] > min_tvl) &
    (data['Spread'] > min_spread)
]

# --- Chart 1: Spread vs Trade Size ---
col6, col7 = st.columns([1, 2])
with col6:
    st.markdown("### 🧭 Slippage vs. Trade Size")
    st.markdown("""
**Purpose**: Evaluate execution quality across different liquidity depths  
**What to look for?** Watch for steep slopes — they signal thin liquidity.
    """)
with col7:
    if not filtered.empty:
        line_data = filtered.groupby(['trade_size_bin', 'pool'])['Spread'].mean().unstack().fillna(0)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        line_data.plot(marker='o', ax=ax1)
        ax1.set_title("Slippage vs. Trade Size")
        ax1.set_ylabel("Spread (%)")
        ax1.grid(True)
        st.pyplot(fig1)
    else:
        st.warning("No data for selected filters.")

# --- Chart 2: Heatmap ---
col8, col9 = st.columns([1, 2])
with col8:
    st.markdown("### 🧊 Slippage Heatmap")
    st.markdown("""
**Purpose**: Uncover execution weaknesses across pools  
**What to look for?** Darker shades indicate worse execution due to liquidity gaps.
    """)
with col9:
    if not filtered.empty:
        heatmap_data = filtered.pivot_table(index="pool", columns="order_size_bin", values="Spread", aggfunc="mean")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".2f", ax=ax2)
        ax2.set_title("Slippage Heatmap (Token vs Order Size)")
        st.pyplot(fig2)
    else:
        st.warning("No data for selected filters.")





# ـــ

st.markdown("---")
st.caption("Contact me: bellabahramii@gmail.com")


