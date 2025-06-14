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
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler

# __________________ Introduction ______________________________________________________________________

data = pd.read_csv('https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/01_Market_Making/df_main_1.csv', on_bad_lines='skip')

st.set_page_config(layout="wide")
st.title("🔍 Stable Pools Market Maker Radar")

# __________________ Filters ______________________________________________________________________

trade_size_order = [
    "All", "≤10k", "10k–50k", "50k–100k", "100k–200k",
    "200k–300k", "300k–400k", "400k–500k",
    "500k–750k", "750k–1M", ">1M"
]

col1, col2, col3, col4, col5 = st.columns(5)
selected_chain = col1.selectbox("Select Blockchain", ["All"] + sorted(data["blockchain"].unique().tolist()))
selected_dex = col2.selectbox("Select DEX", ["All"] + sorted(data["dex"].unique().tolist()))

if selected_dex == "All":
    filtered_pool_options = data["pool"].unique().tolist()
else:
    filtered_pool_options = data[data["dex"] == selected_dex]["pool"].unique().tolist()

selected_pool = col3.selectbox("Select Pool", ["All"] + sorted(filtered_pool_options))
min_trade_size = col4.selectbox("Minimum Trade Size ($)", options=trade_size_order, index=0)
min_spread = col5.number_input("Minimum Spread (%)", value=0)

filtered_data = data.copy()
if selected_chain != "All":
    filtered_data = filtered_data[filtered_data["blockchain"] == selected_chain]

if selected_dex != "All":
    filtered_data = filtered_data[filtered_data["dex"] == selected_dex]

if selected_pool != "All":
    filtered_data = filtered_data[filtered_data["pool"] == selected_pool]


if min_trade_size != "All":
    trade_size_bins = trade_size_order[1:]
    selected_index = trade_size_bins.index(min_trade_size)
    allowed_bins = trade_size_bins[selected_index:]
    filtered_data = filtered_data[filtered_data["trade_size_bin"].isin(allowed_bins)]



filtered_data = filtered_data[filtered_data["Spread"] >= min_spread]



# __________________ Part1: Pie chart + Pool Count  ______________________________________________________________________

grouped = filtered_data.groupby("pool").agg({
    "swap_count": "sum",
    "Trade_size": "mean"
}).reset_index()

grouped.rename(columns={"pool": "pool_name"}, inplace=True)

grouped["swap_share"] = grouped["swap_count"] / grouped["swap_count"].sum()
grouped["trade_size_share"] = grouped["Trade_size"] / grouped["Trade_size"].sum()

fig1 = px.pie(
    grouped,
    values='swap_share',
    names='pool_name',
    title=f"📌 Market Share by Swap Count on {selected_dex} ({selected_chain})",
    hole=0.45
)

fig2 = px.pie(
    grouped,
    values='trade_size_share',
    names='pool_name',
    title=f"📌 Market Share by Trade Size on {selected_dex} ({selected_chain})",
    hole=0.45
)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.plotly_chart(fig2, use_container_width=True)



st.markdown("---")

# __________________ Part2: Trade Size vs. Slippage ______________________________________________________________________

st.subheader("📈 Spread vs. Trade Size")

filtered_data['Spread'] = filtered_data['Spread'] / 100

trade_size_order = [
    "≤10k", "10k–50k", "50k–100k", "100k–200k", "200k–300k", "300k–400k",
    "400k–500k", "500k–750k", "750k–1M", ">1M"
]
col_text1 , col_chart1 = st.columns([1, 1])

with col_chart1:
    line_chart = alt.Chart(filtered_data).mark_line(point=True).encode(
        x=alt.X("trade_size_bin", title="Trade Size (binned)",axis=alt.Axis(labelAngle=30, labelOverlap=False) ,sort=trade_size_order),
        y=alt.Y("Spread", title="Spread (%)"),
        color="pool"
    ).properties(height=400)

    st.altair_chart(line_chart, use_container_width=True)

with col_text1:
    st.markdown("### What to look for?")
    st.markdown("""
    - **Goal**: Assess how trade size affects slippage.
    - **Look for**: Steep slopes, which indicate pools with shallow depth.
    """)


st.markdown("---")

# __________________ Part3: Heatmap ______________________________________________________________________

st.subheader("🔥 Slippage Heatmap (Pool vs Order Size)")

order_size_order = [
    "≤1k", "1k–5k", "5k–10k", "10k–25k", "25k–50k",
    "50k–100k", "100k–250k", "250k–1M", ">1M"
]


col_text2  , col_chart2 = st.columns([1, 1])

with col_chart2:
    heatmap = alt.Chart(filtered_data).mark_rect().encode(
        x=alt.X("order_size_bin", title="Order Size (binned)",axis=alt.Axis(labelAngle=30, labelOverlap=False),sort=order_size_order),
        y=alt.Y("pool", title="Pool"),
        color=alt.Color("Spread:Q", scale=alt.Scale(scheme='redyellowgreen', reverse=True), title="Spread (%)")
    ).properties(height=500)

    st.altair_chart(heatmap, use_container_width=True)

with col_text2:
    st.markdown("### What to look for?")
    st.markdown("""
    - **Goal**: Spot liquidity weakness by order size and pool.
    - **Look for**: Brighter red blocks—higher slippage means riskier for MM.
    """)

st.markdown("---")
# __________________ Part4: Boxplot ______________________________________________________________________

st.subheader("📊 Spread Distribution by DEX")

col_text3, col_chart3 = st.columns([1, 1])

with col_chart3:
    box_plot = alt.Chart(filtered_data).mark_boxplot(extent='min-max').encode(
        x=alt.X("dex:N", title="DEX"),
        y=alt.Y("Spread", title="Spread (%)", scale=alt.Scale(type='log')),
        color="dex:N"
    ).properties(
        height=400,
        width=600
    )
    st.altair_chart(box_plot, use_container_width=True)

with col_text3:
    st.markdown("### What to look for?")
    st.markdown("""
    - **Goal**: Compare how well different DEXs manage slippage (spread).
    - **Look for**: DEXs with higher median or wider spread distribution; these may indicate weaker liquidity providers, offering **market making opportunities**.
    """)

st.markdown("---")
# __________________ Part5: Scoring System ______________________________________________________________________

st.subheader("🧠 Pool Scoring System")
st.markdown("### 🎛️ Adjust Scoring Weights")

col1, col2, col3 = st.columns(3)
with col1:
    volume_w = st.slider("Weight: Volume", 0.0, 1.0, 0.25, 0.05)
    swap_w = st.slider("Weight: Swap Count", 0.0, 1.0, 0.2, 0.05)
with col2:
    order_size_w = st.slider("Weight: Order Size", 0.0, 1.0, 0.1, 0.05)
    trade_size_w = st.slider("Weight: Trade Size", 0.0, 1.0, 0.1, 0.05)
with col3:
    spread_mean_w = st.slider("Weight: Spread (↓ better)", 0.0, 1.0, 0.25, 0.05)
    spread_std_w = st.slider("Weight: Spread Volatility (↓ better)", 0.0, 1.0, 0.1, 0.05)

# Normalize total weight to avoid overweighting
total_weight = volume_w + swap_w + order_size_w + spread_mean_w + spread_std_w + trade_size_w

# --- Score Calculation ---
pool_stats = data.groupby(["blockchain", "dex", "pool"]).agg({
    "volume": "mean",
    "swap_count": "mean",
    "order_size": "mean",
    "Spread": ["mean", "std"],
    "Trade_size": "mean"
})
pool_stats.columns = ['volume_mean', 'swap_count_mean', 'order_size_mean', 'spread_mean', 'spread_std', 'trade_size_mean']
pool_stats = pool_stats.reset_index()

# Normalize
scaler = MinMaxScaler()
metrics = ['volume_mean', 'swap_count_mean', 'order_size_mean', 'spread_mean', 'spread_std', 'trade_size_mean']
pool_stats_normalized = pool_stats.copy()
pool_stats_normalized[metrics] = scaler.fit_transform(pool_stats[metrics])

# MM Score
pool_stats_normalized["mm_score"] = (
    (pool_stats_normalized["volume_mean"] * volume_w) +
    (pool_stats_normalized["swap_count_mean"] * swap_w) +
    (pool_stats_normalized["order_size_mean"] * order_size_w) +
    ((1 - pool_stats_normalized["spread_mean"]) * spread_mean_w) +
    ((1 - pool_stats_normalized["spread_std"]) * spread_std_w) +
    (pool_stats_normalized["trade_size_mean"] * trade_size_w)
) / total_weight

# --- Display Layout ---
st.markdown("### 🔍 Results & Interpretation")
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### ℹ️ How This Scoring Works")
    st.markdown("""
    This scoring system evaluates how suitable a pool is for **market making**.
    
    **Metrics considered:**
    - ✅ High volume, trade size, swap activity
    - ⚠️ Low average spread and spread volatility

    The higher the score, the more promising the pool is for stable MM profits.
    """)

with col_right:
    st.markdown("#### 🏆 Top 20 Pools")
    st.dataframe(pool_stats_normalized.sort_values("mm_score", ascending=False)[
        ["blockchain", "dex", "pool", "mm_score"]
    ].head(20), use_container_width=True)


# ـــ

st.markdown("---")
st.caption("Contact me: bellabahramii@gmail.com")


