#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --------------------------------------
# Load main dataset
df = pd.read_csv(
    'https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/01_Market_Making/Market%20Making.csv',
    on_bad_lines='skip'
)

# --------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Market Making Board")

# _____ Filters _______________________________________
col1, col2, col3 = st.columns(3)
with col1:
    selected_chain = st.selectbox("🔗 Select Chain", sorted(df['Chain'].unique()))
with col2:
    selected_dex = st.selectbox("📈 Select DEX", sorted(df['Dex'].unique()))
with col3:
    selected_type = st.selectbox("💱 Select Pool Type", sorted(df['Pool_Type'].unique()))

filtered_df = df[
    (df['Chain'] == selected_chain) &
    (df['Dex'] == selected_dex)
]

# _____ Pic Chart Market Share _______________________________________

fig1 = px.pie(
    filtered_df,
    values='Share',
    names='Pool_Name',
    title=f"📌 Market Share of Pools on {selected_dex} ({selected_chain})",
    hole=0.45)

# _____ Pic Chart Pool Type _______________________________________

type_counts = filtered_df['Pool_Type'].value_counts(normalize=True) * 100
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

# __________________

df_tvl = pd.read_csv("https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/01_Market_Making/df_tvl.csv")
df_tvl['timestamp'] = pd.to_datetime(df_tvl['timestamp'])

def format_number(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    else:
        return str(n)

dex_options = df["Dex"].unique()
selected_dex = st.multiselect("Select Dex(s):", options=dex_options, default=dex_options)

filtered_dex_pools = df[df["Dex"].isin(selected_dex)][["pool_name", "pool_id", "Dex"]]
df_tvl_filtered = df_tvl.merge(filtered_dex_pools, on="pool_id")

# 📈 TVL Line Chart
st.subheader("📉 TVL Trend Over Time per Dex")

fig, ax = plt.subplots(figsize=(12, 6))
for dex_name in df_tvl_filtered["Dex"].unique():
    data = df_tvl_filtered[df_tvl_filtered["Dex"] == dex_name]
    grouped = data.groupby("timestamp")["tvlUsd"].sum().reset_index()
    ax.plot(grouped["timestamp"], grouped["tvlUsd"], label=dex_name)

ax.set_ylabel("TVL (USD)")
ax.set_xlabel("Date")
ax.legend()
ax.set_title("Total TVL by Dex (Over Time)")
ax.grid(True)
labels = [format_number(y) for y in ax.get_yticks()]
ax.set_yticklabels(labels)

st.pyplot(fig)


# ـــ

st.markdown("---")
st.caption("Contact me: bellabahramii@gmail.com")


