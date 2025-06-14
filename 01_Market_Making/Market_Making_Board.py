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

# __________________ Introduction ______________________________________________________________________

data = pd.read_csv('https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/01_Market_Making/df_main_1.csv', on_bad_lines='skip')

st.set_page_config(layout="wide")
st.title("🔍 Stable Pools Market Maker Radar")

# __________________ Filters ______________________________________________________________________

col1, col2, col3, col4, col5 = st.columns(5)
selected_chain = col1.selectbox("Select Blockchain", ["All"] + sorted(data["blockchain"].unique().tolist()))
selected_dex = col2.selectbox("Select DEX", ["All"] + sorted(data["dex"].unique().tolist()))

if selected_dex == "All":
    filtered_pool_options = data["pool"].unique().tolist()
else:
    filtered_pool_options = data[data["dex"] == selected_dex]["pool"].unique().tolist()

selected_pool = col3.selectbox("Select Pool", ["All"] + sorted(filtered_pool_options))
min_trade_size = col4.number_input("Minimum Trade Size ($)", value=0)
min_spread = col5.number_input("Minimum Spread (%))", value=0)

filtered_data = data.copy()
if selected_chain != "All":
    filtered_data = filtered_data[filtered_data["blockchain"] == selected_chain]

if selected_dex != "All":
    filtered_data = filtered_data[filtered_data["dex"] == selected_dex]

if selected_pool != "All":
    filtered_data = filtered_data[filtered_data["pool"] == selected_pool]

# filtered_data = filtered_data[filtered_data["Trade_size"] >= min_trade_size]
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

col_text1 , col_chart1 = st.columns([1, 1])

with col_chart1:
    line_chart = alt.Chart(filtered_data).mark_line(point=True).encode(
        x=alt.X("trade_size_bin", title="Trade Size (binned)",axis=alt.Axis(labelAngle=30),
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

col_text2  , col_chart2 = st.columns([1, 1])

with col_chart2:
    heatmap = alt.Chart(filtered_data).mark_rect().encode(
        x=alt.X("order_size_bin", title="Order Size (binned)"),
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

# ـــ

st.markdown("---")
st.caption("Contact me: bellabahramii@gmail.com")


