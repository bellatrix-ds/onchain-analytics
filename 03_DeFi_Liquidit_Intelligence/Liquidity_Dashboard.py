import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from matplotlib.dates import DateFormatter
import statsmodels
import altair as alt
from sklearn.preprocessing import MinMaxScaler
import requests
import json
import google.generativeai as genai
from together import Together
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import pipeline
from datetime import datetime
import scipy

# __________________ Import Data ______________________________________________________________________

data = pd.read_csv('https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/03_DeFi_Liquidit_Intelligence/df_main.csv', on_bad_lines='skip')

st.title("🚀 AI‑Powered DeFi Liquidity Intelligence")
st.set_page_config(layout="wide")


st.markdown("""
## Welcome to the AI‑Powered DeFi Liquidity Dashboard 🚀

**Overview:**  
This dashboard automatically fetches on‑chain lending data, visualizes key metrics, and generates AI‑driven insights across four sections.
This dashboard uses **streaming data automation and AI‑powered insight engines** to fetch, analyze, and visualize key metrics for liquidity and risk in lending pools—no manual data wrangling required!

**🛠️ How to use**
**Sections:**  
- **Net Flow Over Time** – Area chart showing deposit vs withdrawal trends  
- **Utilization Rate Over Time** – Line chart tracking pool usage and demand  
- **Core Metrics** (Deposit / Loan / Repay / Withdraw) – Multi-line chart displaying user behavior and liquidity flows  
- **APR vs Utilization Rate** – Scatter plot analyzing interest sensitivity to pool usage  

### 🎯 Final Section: Scenario Insights  
Here, an AI agent reviews the last 30 days of Net Flow, APR, and Utilization, and provides **3 concise bullet-point insights** focused on:
1. Liquidity stress or volatility  
2. Risk alerts related to abnormal outflows  
3. Unusual yield or usage patterns  
""")

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

filtered_data['block_timestamp'] = pd.to_datetime(filtered_data['block_timestamp'], errors='coerce')

st.markdown("___")
# __________________ Part 1: Utilization Rate Over Time ______________________________________________________________________

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
# __________________ Part 2: Net Flow Over Time ______________________________________________________________________

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
# __________________ Part 3: Liquidity Metrics  ______________________________________________________________________
st.markdown("### 💸 Lending Flow Metrics Over Time")

df_amounts = filtered_data[['block_timestamp', 'deposit_amount', 'loan_amount', 'repay_amount', 'withdraw_amount']].copy()
df_amounts['block_timestamp'] = pd.to_datetime(df_amounts['block_timestamp'])

df_melted = df_amounts.melt(
    id_vars='block_timestamp',
    value_vars=['deposit_amount', 'loan_amount', 'repay_amount', 'withdraw_amount'],
    var_name='Type', value_name='Amount'
)

col1, col2 = st.columns([1, 1])

with col1:
    fig = px.line(df_melted, x='block_timestamp', y='Amount', color='Type'
                  , markers=False,
                  color_discrete_map={
                      'deposit_amount': '#4CAF50',
                      'loan_amount': '#2196F3',
                      'repay_amount': '#FFC107',
                      'withdraw_amount': '#F44336'
                  })
    fig.update_layout(height=380, xaxis_title="Date", yaxis_title="Amount (WETH)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 💡 Lending Behavior Insights - AI Generated")

    recent_data = df_amounts.sort_values("block_timestamp").tail(30)
    formatted_data = "\n".join(
        f"{row['block_timestamp'].strftime('%Y-%m-%d')}: deposit={row['deposit_amount']:.2f}, "
        f"loan={row['loan_amount']:.2f}, repay={row['repay_amount']:.2f}, withdraw={row['withdraw_amount']:.2f}"
        for _, row in recent_data.iterrows()
    )

    prompt = (
        "You are a blockchain DeFi analyst focused on lending protocols:\n\n"
        f"{formatted_data}\n\n"
        "Provide 3 concise, smart, non-obvious insights based on this time series in bullet points.Focus on behavior of deposit, loan, repay, and withdraw, below there are daily metrics related to deposit, loan, repay, and withdraw daily"
    )

    headers = {
        "Authorization": f"Bearer {st.secrets['TOGETHER_API_KEY']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a professional DeFi analyst."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 200
    }

    try:
        response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)
        result = response.json()
        content = result['choices'][0]['message']['content']
        for line in content.strip().split('\n'):
            if line.strip():
                st.write(f"• {line.strip().lstrip('-•')}")
    except Exception as e:
        st.error(f"AI insight error: {e}")



st.markdown("___")
# __________________ Part 4: Scatter Plot APR vs. Utilization Rate ______________________________________________________________________


df_scatter = filtered_data[["APR", "utilization_rate"]].dropna()
df_scatter = df_scatter[df_scatter["utilization_rate"] > 0]

st.markdown("### 📌 APR vs Utilization Rate")

col10, col20 = st.columns(2)

with col10:
    fig = px.scatter(
        df_scatter,
        x="utilization_rate",
        y="APR",
        title="APR vs Utilization Rate",
        labels={"APR": "Annual Percentage Rate", "utilization_rate": "Utilization Rate"},
        color_discrete_sequence=["#FF5733"]
    )
    fig.update_layout(height=370)
    st.plotly_chart(fig, use_container_width=True)

with col20:
    st.markdown("#### ✅ Lending Efficiency Insights")

    sample_rows = df_scatter.tail(30)
    data_summary = "\n".join(f"{r['utilization_rate']:.2f}, {r['APR']:.2f}" for _, r in sample_rows.iterrows())

    prompt = (
        f"Here are 30 recent (utilization_rate, APR) observations from a DeFi lending pool:\n"
        f"{data_summary}\n\n"
        f"Please summarize the relationship between APR and utilization in 3 short bullet points."
    )

    together_api_key = st.secrets["TOGETHER_API_KEY"]
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a DeFi lending analyst."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 256
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        output = result['choices'][0]['message']['content']
        for bullet in output.strip().split("\n"):
            if bullet.strip():
                st.write(f"• {bullet.strip().lstrip('-•')}")
    except Exception as e:
        st.error(f"AI Insight error: {e}")


st.markdown("___")
# __________________ Part 5 ______________________________________________________________________

scenarios = {
    "📉 Decreasing Net Flow": "Analyze signs of capital outflows and what risks it may imply for the lending pool.",
    "📈 Increasing Utilization": "What does a rising utilization rate mean for APR and borrower demand?",
    "🔁 Sudden APR Changes": "Explain the potential causes and implications of volatile APR behavior.",
    "🧊 Zero Borrow Activity": "Interpret days with zero utilization and what they signal for market sentiment.",
    "🔥 Liquidity Crunch": "Could recent data suggest a liquidity crisis or risk of insolvency?" }

st.markdown("### 🧪 Scenario-based Insight Generator")
selected_scenario = st.radio("Choose a scenario to analyze:", list(scenarios.keys()))
scenario_instruction = scenarios[selected_scenario]

context_data = filtered_data[['block_timestamp', 'net_flow', 'APR', 'utilization_rate']].dropna().tail(30)
context_data['block_timestamp'] = pd.to_datetime(context_data['block_timestamp'])
summary = "\n".join(
    f"{r['block_timestamp'].strftime('%d %b %Y')} | Net Flow: {r['net_flow']:.2f}, APR: {r['APR']:.2f}, Util: {r['utilization_rate']:.2f}"
    for _, r in context_data.iterrows()
)

prompt = f"""
You are a DeFi protocol analyst. Below is recent daily data from a lending pool on a blockchain:

Date | Net Flow | APR | Utilization Rate
{summary}

Scenario: {scenario_instruction}

Give 3 bullet-point insights for this scenario based on the above data.
"""

together_api_key = st.secrets["TOGETHER_API_KEY"]
url = "https://api.together.xyz/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {together_api_key}",
    "Content-Type": "application/json"
}
payload = {
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "messages": [
        {"role": "system", "content": "You are a professional DeFi protocol analyst."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.6,
    "top_p": 0.9,
    "max_tokens": 600
}

col1, col2 = st.columns(2)

with col1:
    st.markdown("📊 Last 30 Days Data")
    st.dataframe(context_data)

with col2:
    st.markdown("#### 🟢 Scenario-based Insight")
    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        output = result['choices'][0]['message']['content']
        for line in output.strip().split("\n"):
            if line.strip():
                st.write("• " + line.strip().lstrip("-•"))
    except Exception as e:
        st.error(f"AI Insight error: {e}")
