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
    'https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/01_Market_Making/test.csv',
    on_bad_lines='skip'
)

# --------------------------------------
st.set_page_config(page_title="Spread vs TVL", layout="centered")
st.title("📈 Spread vs TVL (Curve Pools on Ethereum)")
st.markdown("نقاط بالا-سمت‌چپ → اسپرد زیاد ولی TVL کم → فرصت برای market maker")

df['SPREAD_PERC'] = pd.to_numeric(df['SPREAD_PERC'], errors='coerce')
df['VOLUME_USD'] = pd.to_numeric(df['VOLUME_USD'], errors='coerce')
df = df.dropna(subset=['SPREAD_PERC', 'VOLUME_USD'])

# رسم نمودار
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['VOLUME_USD'], df['SPREAD_PERC'], alpha=0.7, color='tomato')
ax.set_xscale('log')
ax.set_xlabel("TVL (Volume USD, log scale)")
ax.set_ylabel("Spread (%)")
ax.set_title("Spread vs TVL Scatter Plot")
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# نمایش اسم pool برای نقاط جالب (اسپرد بالا + TVL پایین)
for _, row in df.iterrows():
    if row['SPREAD_PERC'] > 0.25 and row['VOLUME_USD'] < 1e7:
        ax.text(row['VOLUME_USD'], row['SPREAD_PERC'], row['POOL'], fontsize=7)

# خروجی در Streamlit
st.pyplot(fig)

# ـــ

st.markdown("---")
st.caption("Contact me: bellabahramii@gmail.com")


