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
# Load main dataset
df = pd.read_csv(
    'https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/01_Market_Making/Market%20Making.csv',
    on_bad_lines='skip'
)
print(df.columns)
# --------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Market Making Board")

# _____ Filters _______________________________________
col1, col2, col3 = st.columns(3)
with col1:
    selected_chain = st.selectbox("🔗 Select chain", sorted(df['chain'].unique()))
with col2:
    selected_dex = st.selectbox("📈 Select DEX", sorted(df['dex'].unique()))
with col3:
    selected_type = st.selectbox("💱 Select Pool Type", sorted(df['pool_type'].unique()))

filtered_df = df[
    (df['chain'] == selected_chain) &
    (df['dex'] == selected_dex)
]

# _____ Pic Chart Market Share _______________________________________

fig1 = px.pie(
    filtered_df,
    values='Share',
    names='pool_name',
    title=f"📌 Market Share of Pools on {selected_dex} ({selected_chain})",
    hole=0.45)

# _____ Pic Chart Pool Type _______________________________________

type_counts = filtered_df['pool_type'].value_counts(normalize=True) * 100
type_df = pd.DataFrame({
    'Type': type_counts.index,
    'Percentage': type_counts.values
})

fig2 = px.pie(
    type_df,
    names='Type',
    values='Percentage',
    title=f"🔍 Pool Type Distribution on {selected_dex}",
    hole=0.45
)

col4, col5 = st.columns(2)
with col4:
    st.plotly_chart(fig1, use_container_width=True)
with col5:
    st.plotly_chart(fig2, use_container_width=True)


# _____ Table _______________________________________

st.subheader("📄 Filtered Pool Table")
st.dataframe(filtered_df.reset_index(drop=True))

# __________________ Part 4 ______________________________________________________________________

df_tvl = pd.read_csv("https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/01_Market_Making/df_tvl.csv")
df_tvl['timestamp'] = pd.to_datetime(df_tvl['timestamp'])

def format_number(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.0f}M"  # بدون اعشار
    elif n >= 1e3:
        return f"{n / 1e3:.0f}K"
    else:
        return str(n)

# Filter DEX selection
dex_options = df["dex"].unique()
selected_dex = st.multiselect("Select Dex(s):", options=dex_options, default=dex_options)

# Merge and filter
filtered_dex_pools = df[df["dex"].isin(selected_dex)][["pool_id", "dex"]]
df_tvl_filtered = df_tvl.merge(filtered_dex_pools, on="pool_id", how="inner")

# Line chart section
st.subheader("📉 4. TVL Trend Over Time per Dex")

col1, col2 = st.columns([1.2, 2], gap="large")

with col1:
    st.markdown(
                """
        <div style='margin-top: 25px'>
        🔍 <b>What to look for?</b><br>
        In this chart, we're tracking the market's confidence in DEX liquidity pools.<br>
        Pay attention to sharp drops; they might indicate a risk of capital flight.<br>
        Rising TVL trends suggest growing trust and increasing capital allocation.
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    fig, ax = plt.subplots(figsize=(10, 5))
    for pool in df_tvl_filtered["pool_name"].unique():
        data = df_tvl_filtered[df_tvl_filtered["pool_name"] == pool]
        grouped = data.groupby("timestamp")["tvlUsd"].sum().reset_index()
        ax.plot(grouped["timestamp"], grouped["tvlUsd"], label=pool)

    ax.set_ylabel("TVL (in millions USD)")
    ax.set_xlabel("Date")
    ax.legend(    title="Pool Name",
    bbox_to_anchor=(1.05, 1), 
    loc='upper left',
    fontsize='small'
)
    ax.set_title("Total TVL by Dex (Over Time) — 2025")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
    ax.xaxis.set_major_formatter(DateFormatter('%b-%d'))

    ax.grid(False)
    st.pyplot(fig)







# ـــ

st.markdown("---")
st.caption("Contact me: bellabahramii@gmail.com")


