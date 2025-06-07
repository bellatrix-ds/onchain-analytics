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
df['SPREAD_PERC'] = pd.to_numeric(df['SPREAD_PERC'], errors='coerce')
df['VOLUME_USD'] = pd.to_numeric(df['VOLUME_USD'], errors='coerce')

plt.figure(figsize=(10, 6))
plt.scatter(df['VOLUME_USD'], df['SPREAD_PERC'], alpha=0.7, c='tomato')
plt.xscale("log")  
plt.xlabel("TVL (Volume USD - log scale)")
plt.ylabel("Spread (%)")
plt.title("📈 Spread vs TVL (Curve Pools - Ethereum)")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

# ـــ

st.markdown("---")
st.caption("Contact me: bellabahramii@gmail.com")


