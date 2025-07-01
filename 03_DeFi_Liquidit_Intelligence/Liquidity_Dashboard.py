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
import requests
import json
import google.generativeai as genai

# __________________ Import Data ______________________________________________________________________

data = pd.read_csv('https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/03_DeFi_Liquidit_Intelligence/df_main.csv', on_bad_lines='skip')

st.title("🔍 Pool Liquidity Intelligence")


# __________________ Filters ______________________________________________________________________

# Lending Protocol
lending_protocols = data['lending_protocol'].dropna().unique()
selected_protocol = st.selectbox("Select Lending Protocol", sorted(lending_protocols))

# Chain
filtered_chains = data[data['lending_protocol'] == selected_protocol]['chain'].dropna().unique()
selected_chain = st.selectbox("Select Chain", sorted(filtered_chains))

# Pool
filtered_pools = data[
    (data['lending_protocol'] == selected_protocol) &
    (data['chain'] == selected_chain)
]['pool'].dropna().unique()
selected_pool = st.selectbox("Select Pool", sorted(filtered_pools))

filtered_df = data[
    (data['lending_protocol'] == selected_protocol) &
    (data['chain'] == selected_chain) &
    (data['pool'] == selected_pool)
]


# __________________ Part 1: Trends ______________________________________________________________________

col1, col2 = st.columns(2)

with col1:
    df_plot = filtered_df[['block_timestamp', 'utilization_rate']].dropna()
    df_plot['block_timestamp'] = pd.to_datetime(df_plot['block_timestamp'])

    fig = px.line(df_plot, x='block_timestamp', y='utilization_rate',
                  title='Utilization Rate Over Time', markers=True)
    fig.update_layout(xaxis_title='Date', yaxis_title='Utilization Rate', height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### ℹ️ توضیح")
    st.write("""
    این نمودار میزان استفاده از نقدینگی را در طول زمان برای استخر انتخاب‌شده نشان می‌دهد.  
    نرخ utilization بالا می‌تواند نشانه فشار نقدینگی باشد، در حالی که نرخ پایین ممکن است به معنای سرمایه‌های بلااستفاده باشد.  
    نقاط صفر یا نال معمولاً به معنی فقدان فعالیت یا ثبت‌نشدن داده در آن روز خاص هستند.
    """)
