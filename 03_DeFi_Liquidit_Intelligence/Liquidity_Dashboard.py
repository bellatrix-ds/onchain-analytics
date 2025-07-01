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
from together import Together
import pandas as pd
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import pipeline

# __________________ Import Data ______________________________________________________________________

data = pd.read_csv('https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/03_DeFi_Liquidit_Intelligence/df_main.csv', on_bad_lines='skip')

st.title("🔍 Pool Liquidity Intelligence")
st.set_page_config(layout="wide")

# __________________ Filters ______________________________________________________________________


col1, col2, col3 = st.columns(3)
with col1:
    selected_protocol = st.selectbox("Lending Protocol", options=data['lending_protocol'].dropna().unique())
with col2:
    selected_chain = st.selectbox("Chain", options=data[data['lending_protocol'] == selected_protocol]['chain'].dropna().unique())
with col3:
    selected_pool = st.selectbox("Pool", options=data[(data['lending_protocol'] == selected_protocol) & (data['chain'] == selected_chain)]['pool'].dropna().unique())

filtered_data = data[
    (data['lending_protocol'] == selected_protocol) &
    (data['chain'] == selected_chain) &
    (data['pool'] == selected_pool)
].copy()

#filtered_data['block_timestamp'] = pd.to_datetime(filtered_data['block_timestamp'], errors='coerce')
st.markdown("___")
# __________________ Part 1: Trends ______________________________________________________________________

col4, col5 = st.columns(2)

with col4:
    st.subheader("Utilization Rate Over Time")
    fig = px.line(
        filtered_data,
        x='block_timestamp',
        y='utilization_rate',
        markers=True,
        labels={'block_timestamp': 'Date', 'utilization_rate': 'Utilization Rate'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col5:
    st.markdown("### 🔄 Net Flow یعنی چه؟")
    st.write("""
    این نمودار میزان استفاده از نقدینگی را در طول زمان برای استخر انتخاب‌شده نشان می‌دهد.  
    **بالا** می‌تواند نشانه فشار نقدینگی باشد. در حالی که نرخ **پایین** ممکن است به معنی سرمایه‌های بلااستفاده باشد.  
    نقاط **صفر** یا **نال** معمولاً به معنی فقدان فعالیت یا ثبت‌نشدن داده در آن روز خاص هستند.
    """)

st.markdown("___")
# __________________ Part 2: Net Flow ______________________________________________________________________

# Net Flow chart
fig = px.area(filtered_data, x='block_timestamp', y='net_flow', title="Net Flow Over Time", color_discrete_sequence=['#2196F3'])
fig.update_layout(xaxis_title='Date', yaxis_title='Net Flow', height=350)
st.plotly_chart(fig, use_container_width=True)

# AI analysis
col1, col2 = st.columns([1, 2])

with col1:
    st.write("")

with col2:
    st.markdown("### 🤖 AI Insight")

    try:
        from transformers import pipeline

        @st.cache_resource
        def load_pipe():
            return pipeline("text2text-generation", model="google/flan-t5-base")

        pipe = load_pipe()

        prompt = (
            "The following is the daily net flow (deposit - withdraw) for a DeFi lending pool:\n"
            + "\n".join(
                f"{row['block_timestamp'].date()}: {row['net_flow']:.2f}"
                for _, row in filtered_data[['block_timestamp', 'net_flow']]
                .dropna()
                .sort_values('block_timestamp')
                .iterrows()
            )
            + "\n\nGive 3 concise, smart insights in bullet points."
        )

        result = pipe(prompt, max_new_tokens=256)[0]['generated_text']

        for line in result.split("\n"):
            if '-' in line or '•' in line:
                st.write(line.strip().lstrip("-•"))

    except Exception as e:
        st.error(f"AI insight error: {e}")

    except Exception as e:
        st.error(f"AI insight error: {e}")

st.markdown("___")
