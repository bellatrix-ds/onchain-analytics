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
# --------------------------------------


# __________________ New ______________________________________________________________________


data = pd.read_csv('https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/01_Market_Making/df_main.csv', on_bad_lines='skip')

st.set_page_config(layout="wide")

# trade size
data["trade_size"] = data["volume"] / data["swap_count"]

# باین کردن trade_size و order_size
data["trade_size_bin"] = pd.cut(
    data["trade_size"], 
    bins=[0, 10_000, 50_000, 100_000, 500_000, 1_000_000, float("inf")],
    labels=["<10k", "10k-50k", "50k-100k", "100k-500k", "500k-1M", "1M+"]
)

data["order_size_bin"] = pd.cut(
    data["order_size"], 
    bins=[0, 10_000, 50_000, 100_000, 500_000, 1_000_000, float("inf")],
    labels=["<10k", "10k-50k", "50k-100k", "100k-500k", "500k-1M", "1M+"]
)

# عنوان
 
st.title("🔍 Stable Pools Market Maker Radar")

# فیلترها (بالاتر از نمودارها)
col1, col2, col3, col4, col5 = st.columns(5)

# Default selections = all
selected_chain = col1.selectbox("Select Blockchain", ["All"] + sorted(data["blockchain"].unique().tolist()))
selected_dex = col2.selectbox("Select DEX", ["All"] + sorted(data["dex"].unique().tolist()))

# فیلتر وابسته برای pool
if selected_dex == "All":
    filtered_pool_options = data["pool"].unique().tolist()
else:
    filtered_pool_options = data[data["dex"] == selected_dex]["pool"].unique().tolist()

selected_pool = col3.selectbox("Select Pool", ["All"] + sorted(filtered_pool_options))
min_tvl = col4.number_input("Minimum TVL ($)", value=0)
min_spread = col5.slider("Minimum Spread (%)", 0.0, 5.0, 0.0, step=0.1)

# اعمال فیلترها روی دیتافریم
filtered_data = data.copy()
if selected_chain != "All":
    filtered_data = filtered_data[filtered_data["blockchain"] == selected_chain]

if selected_dex != "All":
    filtered_data = filtered_data[filtered_data["dex"] == selected_dex]

if selected_pool != "All":
    filtered_data = filtered_data[filtered_data["pool"] == selected_pool]

# filtered_data = filtered_data[filtered_data["tvl_usd"] >= min_tvl]
filtered_data = filtered_data[filtered_data["Spread"] >= min_spread]

# بخش اول: Trade Size vs. Slippage
st.subheader("📈 Spread vs. Trade Size")

col_text1 , col_chart1 = st.columns([1, 1])

with col_chart1:
    line_chart = alt.Chart(filtered_data).mark_line(point=True).encode(
        x=alt.X("trade_size_bin:N", title="Trade Size (binned)"),
        y=alt.Y("Spread:Q", title="Spread (%)"),
        color="pool:N"
    ).properties(height=400)

    st.altair_chart(line_chart, use_container_width=True)

with col_text1:
    st.markdown("### What to look for?")
    st.markdown("""
    - **Goal**: Assess how trade size affects slippage.
    - **Look for**: Steep slopes, which indicate pools with shallow depth.
    """)

# بخش دوم: Heatmap
st.subheader("🔥 Slippage Heatmap (Pool vs Order Size)")

col_text2  , col_chart2 = st.columns([1, 2])

with col_chart2:
    heatmap = alt.Chart(filtered_data).mark_rect().encode(
        x=alt.X("order_size_bin:N", title="Order Size (binned)"),
        y=alt.Y("pool:N", title="Pool"),
        color=alt.Color("Spread:Q", scale=alt.Scale(scheme='redyellowgreen', reverse=True), title="Spread (%)")
    ).properties(height=500)

    st.altair_chart(heatmap, use_container_width=True)

with col_text2:
    st.markdown("### What to look for?")
    st.markdown("""
    - **Goal**: Spot liquidity weakness by order size and pool.
    - **Look for**: Brighter red blocks—higher slippage means riskier for MM.
    """)



# ـــ

st.markdown("---")
st.caption("Contact me: bellabahramii@gmail.com")


