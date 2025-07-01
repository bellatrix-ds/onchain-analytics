import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from matplotlib.dates import DateFormatter

import altair as alt
from sklearn.preprocessing import MinMaxScaler
import requests
import json
import google.generativeai as genai
from together import Together
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import pipeline
from transformers import pipeline
from datetime import datetime

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

df_util = filtered_data[['block_timestamp', 'utilization_rate']].dropna()
df_util['block_timestamp'] = pd.to_datetime(df_util['block_timestamp'])
df_util = df_util.sort_values('block_timestamp')

df_util_recent = df_util.tail(30)

util_col1, util_col2 = st.columns([1.1, 1])

with util_col1:
    st.markdown("#### 📈 Utilization Rate Over Time")
    fig_util = px.line(
        df_util_recent,
        x='block_timestamp',
        y='utilization_rate',
        markers=True,
        color_discrete_sequence=['#FF5722']
    )
    fig_util.update_layout(
        xaxis_title='Date',
        yaxis_title='Utilization Rate',
        height=360,
        yaxis_range=[0, 1.05]
    )
    st.plotly_chart(fig_util, use_container_width=True)

with util_col2:
    st.markdown("#### 💡 AI Generated Insight")

    util_prompt_data = "\n".join(
        f"{row['block_timestamp'].strftime('%Y-%m-%d')}: {row['utilization_rate']:.2f}"
        for _, row in df_util_recent.iterrows()
    )

    util_prompt = (
        "You are a blockchain DeFi analyst focused on lending protocols. "
        "Below is the daily utilization rate (borrowed / total liquidity) of a lending pool. "
        "Provide 3 concise, smart, non-obvious insights based on this time series for each bullet, use each with a relevant emoji (📉, 💡, ⚠️, 🔁):\n\n"
        + util_prompt_data
        + "\n\nFocus on identifying signs of lending demand shifts, liquidity pressure, or inactivity."
    )

    try:
        together_api_key = st.secrets["TOGETHER_API_KEY"]
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {together_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [
                {"role": "system", "content": "You are an expert DeFi lending analyst. Your job is to explain utilization rate trends in lending pools."},
                {"role": "user", "content": util_prompt}
            ],
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 400
        }

        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        output = result['choices'][0]['message']['content']
        for line in output.strip().split('\n'):
            if line.strip():
                st.write(f"• {line.strip().lstrip('-•')}")
    except Exception as e:
        st.error(f"AI insight error: {e}")



st.markdown("___")
# __________________ Part 2: Net Flow ______________________________________________________________________

df_netflow = filtered_data[['block_timestamp', 'net_flow']].dropna()
df_netflow['block_timestamp'] = pd.to_datetime(df_netflow['block_timestamp'])

header_col1, header_col2 = st.columns([1.1, 1])

with header_col1:
    st.markdown("#### 📉 Net Flow Over Time")

insight_types = {
    "📈 Trend Analysis": "Analyze the overall trend of net flows in the lending pool.",
    "⚠️ Risk Alerts": "Identify large outflows that may indicate liquidity risks.",
    "📊 Volatility Summary": "Summarize spikes, dips, and variability in net flows.",
    "🔍 Unusual Patterns": "Find 3 interesting or unexpected behaviors in the data."
}

with header_col2:
    selected_type = st.radio("**🔴 Select Insight Type**", list(insight_types.keys()), horizontal=True)

main_col1, main_col2 = st.columns([1.1, 1])

with main_col1:
    fig = px.area(
        df_netflow,
        x='block_timestamp',
        y='net_flow',
        color_discrete_sequence=['#2196F3']
    )
    fig.update_layout(xaxis_title='Date', yaxis_title='Net Flow', height=360)
    st.plotly_chart(fig, use_container_width=True)

with main_col2:
    st.markdown("#### 💡 AI Generated Insight")

    recent_data = df_netflow.sort_values('block_timestamp').tail(30)
    prompt_data = "\n".join(
        f"{row['block_timestamp'].strftime('%Y-%m-%d')}: {row['net_flow']:.2f}"
        for _, row in recent_data.iterrows()
    )

    prompt = (
        f"You are a DeFi analyst. Your task is: {insight_types[selected_type]}\n"
        f"Here is the latest daily Net Flow data for the lending pool:\n\n"
        f"{prompt_data}\n\n"
        f"Now provide 3 concise, non-obvious bullet point insights."
    )

    # 🔐 Together API Call
    try:
        together_api_key = st.secrets["TOGETHER_API_KEY"]
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {together_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [
                {"role": "system", "content": "You are a professional DeFi analyst who gives smart, short, bullet-point insights based on net flow trends."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 200
        }

        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        for line in content.strip().split('\n'):
            if line.strip():
                st.write(f"• {line.strip().lstrip('-•')}")
    except Exception as e:
        st.error(f"AI Insight error: {e}")


st.markdown("___")
# __________________ Part 3: Net Flow ______________________________________________________________________

