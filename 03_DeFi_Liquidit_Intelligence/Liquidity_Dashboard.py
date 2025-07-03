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

st.title("🚀 Welcome to the AI‑Powered DeFi Liquidity Dashboard")
st.set_page_config(layout="wide")


st.markdown("""
This dashboard is fully powered by **raw on‑chain lending data** and **AI‑driven insight engines**.  
It automatically fetches blockchain logs, analyzes key metrics, and visualizes liquidity and risk patterns, no manual data wrangling needed. """)
 

st.image("defi_analysis.png", use_column_width=True)

st.markdown("""
##### 🚀 All set?! Let’s get started! 👊🏼 """)


st.markdown("___")
# __________________ Filters ______________________________________________________________________


col1, col2, col3 = st.columns(3)
with col1:
    selected_protocol = st.selectbox("Select Lending Protocol", options=data['lending_protocol'].dropna().unique())
with col2:
    selected_chain = st.selectbox("Select Chain", options=data[data['lending_protocol'] == selected_protocol]['chain'].dropna().unique())
with col3:
    selected_pool = st.selectbox("Select Pool", options=data[(data['lending_protocol'] == selected_protocol) & (data['chain'] == selected_chain)]['pool'].dropna().unique())

filtered_data = data[
    (data['lending_protocol'] == selected_protocol) &
    (data['chain'] == selected_chain) &
    (data['pool'] == selected_pool)
].copy()

filtered_data['block_timestamp'] = pd.to_datetime(filtered_data['block_timestamp'], errors='coerce')

# __________________ Part 1: Utilization Rate Over Time ______________________________________________________________________

st.markdown("  ")

st.markdown("  ")

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
     "You are a DeFi analyst. Below is the daily utilization rate for a lending pool.\n"
    "Generate 3 concise bullet-point insights. For each:\n"
    "1. Start with a **short question** related to what the trend shows.\n"
    "2. Then provide a **brief answer** explaining the behavior (max 2 short sentences).\n"
    "3. Use a relevant emoji at the start of each answer.\n"
    "Avoid generalities—be direct and insightful.\n\n"
    + util_prompt_data
    )

    try:
        groq_api_key = st.secrets["GROQ_API_KEY"] 
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are an expert DeFi analyst. Your job is to explain utilization rate trends in lending pools."},
                {"role": "user", "content": util_prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 500,
            "top_p": 0.9
        }

        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            output = result["choices"][0]["message"]["content"]
            for line in output.strip().split('\n'):
                if line.strip():
                    st.write(f"• {line.strip().lstrip('-•')}")
        else:
            st.warning("💭 Wait ... I'm thinking! 🤔")

    except Exception as e:
        st.error(f"AI insight error: {e}")


st.markdown("___")
st.markdown("   ")



# __________________ Part 2: Net Flow Over Time ______________________________________________________________________

# Prepare Net Flow data
df_netflow = filtered_data[['block_timestamp', 'net_flow']].dropna()
df_netflow['block_timestamp'] = pd.to_datetime(df_netflow['block_timestamp'])

# Header layout
header_col1, header_col2 = st.columns([1.1, 1])

with header_col1:
    st.markdown("#### 📉 Net Flow Over Time")

# Insight type selection
insight_types = {
    "📈 Trend Analysis": "Identify trends, shifts in behavior, or sustained movements in net flow.",
    "⚠️ Risk Alerts": "Detect signals of high outflows, liquidity risk, or capital flight.",
    "📊 Volatility Summary": "Summarize the volatility or consistency in net flow over time.",
    "🔍 Unusual Patterns": "Spot unusual, non-random behavior like sharp reversals, spikes, or anomalies."
}

with header_col2:
    selected_type = st.radio("**🔴 Select Insight Type**", list(insight_types.keys()), horizontal=True)

# Chart + AI layout
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
    st.markdown("#### 💡 AI Net Flow Insight")

    recent_data = df_netflow.sort_values('block_timestamp').tail(30)
    prompt_data = "\n".join(
        f"{row['block_timestamp'].strftime('%Y-%m-%d')}: {row['net_flow']:.2f}"
        for _, row in recent_data.iterrows()
    )

    insight_instruction = insight_types.get(selected_type, insight_types["📈 Trend Analysis"])

    netflow_prompt = (
        "You are a DeFi analyst. Below is the daily net flow (deposits - withdrawals) for a lending pool.\n"
        f"Your task is: {insight_instruction}\n\n"
        "Generate 3 concise bullet-point insights. For each:\n"
        "1. Start with a **short question** related to the net flow trend.\n"
        "2. Then provide a **brief answer** explaining the behavior (max 2 short sentences).\n"
        "3. Use a relevant emoji at the start of each answer (📉, ⚠️, 💡, etc).\n"
        "Avoid generalities—be direct and insightful.\n\n"
        + prompt_data
    )

    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional DeFi analyst. Your job is to explain net flow trends in lending pools."
                },
                {
                    "role": "user",
                    "content": netflow_prompt
                }
            ],
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 500
        }

        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            output = result["choices"][0]["message"]["content"]
            for line in output.strip().split('\n'):
                if line.strip():
                    st.write(f"• {line.strip().lstrip('-•')}")
        else:
            st.warning("💭 Wait ... I'm thinking! 🤔")
    except Exception as e:
        st.error(f"AI insight error: {e}")
 

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

col1, col2 = st.columns([1.1, 1])

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
    st.markdown("### 💡 Lending Behavior Insights")

    recent_data = df_amounts.sort_values("block_timestamp").tail(30)
    formatted_data = "\n".join(
        f"{row['block_timestamp'].strftime('%Y-%m-%d')}: deposit={row['deposit_amount']:.2f}, "
        f"loan={row['loan_amount']:.2f}, repay={row['repay_amount']:.2f}, withdraw={row['withdraw_amount']:.2f}"
        for _, row in recent_data.iterrows()
    )

    lending_prompt = (
        "You are a DeFi analyst. Below is the daily behavior of a lending pool:\n"
        "Metrics include deposit, loan, repay, and withdraw amounts.\n"
        "Generate 3 smart, concise insights. For each:\n"
        "1. Start with a **short question** about the behavior.\n"
        "2. Then provide a brief, insightful answer (max 2 sentences).\n"
        "3. Use a relevant emoji at the start (📥, 📤, 📉, 🔄, 💰, etc).\n"
        "Avoid generic observations.\n\n"
        + formatted_data
    )

    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional DeFi analyst. Your job is to explain lending behavior patterns across deposits, loans, repayments, and withdrawals."
                },
                {
                    "role": "user",
                    "content": lending_prompt
                }
            ],
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 500
        }

        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            output = result["choices"][0]["message"]["content"]
            for line in output.strip().split('\n'):
                if line.strip():
                    st.write(f"• {line.strip().lstrip('-•')}")
        else:
            st.warning("💭 Wait ... I'm thinking! 🤔")
            st.caption("Model responded but without useful content.")
    except Exception as e:
        st.error("AI insight error")
        st.caption(f"Exception: {e}")


st.markdown("___")
# __________________ Part 4: Scatter Plot APR vs. Utilization Rate ______________________________________________________________________


df_scatter = filtered_data[["APR", "utilization_rate","block_timestamp"]].dropna()
df_scatter = df_scatter[df_scatter["utilization_rate"] > 0]
df_scatter['block_timestamp'] = pd.to_datetime(df_scatter['block_timestamp'])


st.markdown("### 📌 APR vs Utilization Rate")

col10, col20 = st.columns([1.1, 1])

with col10:
    df_scatter["utilization_percent"] = df_scatter["utilization_rate"] * 100  # convert to %
    fig = px.scatter(
        df_scatter,
        x="utilization_percent",
        y="APR",
        labels={"APR": "Annual Percentage Rate", "utilization_percent": "Utilization Rate (%)"},
        color_discrete_sequence=["#FF5733"]
    )
    fig.update_layout(height=370)
    st.plotly_chart(fig, use_container_width=True)

with col20:
    st.markdown("#### ✅ Lending Efficiency Insights")

    df_scatter["block_timestamp"] = pd.to_datetime(df_scatter["block_timestamp"])
    sample_rows = df_scatter.dropna(subset=["block_timestamp", "utilization_rate", "APR"]).tail(30)

    data_summary = "\n".join(
        f"{r['block_timestamp'].strftime('%Y-%m-%d')}: utilization={r['utilization_rate']:.2f}, APR={r['APR']:.2f}"
        for _, r in sample_rows.iterrows()
    )

    efficiency_prompt = (
        "You are a DeFi analyst evaluating lending pool efficiency.\n"
        "Below is the recent daily relationship between utilization rate and APR.\n"
        "Analyze and summarize what the pattern suggests in 3 smart bullet points. For each:\n"
        "1. Start with a **short question** about efficiency or behavior.\n"
        "2. Then give a clear, short answer (max 2 short sentences).\n"
        "3. Begin the answer with a relevant emoji (📈, ⚠️, 💰, 🔄, etc).\n"
        "Avoid generic statements. Focus on insightful relationships.\n\n"
        + data_summary
    )

    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional DeFi analyst. You explain how utilization rate and APR interact in lending pools."
                },
                {
                    "role": "user",
                    "content": efficiency_prompt
                }
            ],
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 500
        }

        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            output = result["choices"][0]["message"]["content"]
            for bullet in output.strip().split("\n"):
                if bullet.strip():
                    st.write(f"• {bullet.strip().lstrip('-•')}")
        else:
            st.warning("💭 Wait ... I'm thinking! 🤔")
            st.caption("The model responded with no usable content.")
    except Exception as e:
        st.error("AI Insight error.")
        st.caption(f"Exception: {e}")

st.markdown("___")
# __________________ Part 5 ______________________________________________________________________

st.markdown("### 🧪 Scenario-based Insight Generator")

scenario_options = [
    ("📉 Decreasing Net Flow", "More funds are leaving the pool than entering — a potential sign of capital flight."),
    ("📈 Increasing Utilization", "Lending demand is rising, possibly reducing idle liquidity and increasing risk."),
    ("🔁 Sudden APR Changes", "The lending rate has changed sharply — may indicate protocol adjustments or volatility."),
    ("🧊 Zero Borrow Activity", "No loans are being taken — could suggest lack of demand or overly high borrowing costs."),
    ("🔥 Liquidity Crunch", "Multiple stress signals suggest borrowers may face trouble getting funds.")
]

selected = st.radio("❗ Choose a scenario to analyze:", [s[0] for s in scenario_options])
scenario_instruction = dict(scenario_options)[selected]

# Prepare context
filtered_data["utilization %"] = filtered_data["utilization_rate"] * 100 
filtered_data["APR %"] = filtered_data["APR"]

context_data = filtered_data[['block_timestamp', 'net_flow', 'APR %', 'utilization %']].dropna().tail(30).copy()
context_data['block_timestamp'] = pd.to_datetime(context_data['block_timestamp']).dt.strftime('%d %b %Y')

summary = "\n".join(
    f"{r['block_timestamp']} | Net Flow: {r['net_flow']:.2f}, APR: {r['APR %']:.2f}, Util: {r['utilization %']:.2f}"
    for _, r in context_data.iterrows()
)

prompt = f"""
You are a senior DeFi lending analyst. Below is recent daily data from a blockchain lending pool:

Date | Net Flow | APR | Utilization Rate
{summary}

Scenario: {scenario_instruction}

Generate 3 short, smart insights in bullet points. For each:
• Start with a **question** that the scenario raises
• Give a **brief answer** (max 2 short sentences)
• Start the answer with a relevant emoji (⚠️, 📉, 🔄, 💡, etc)
"""

# API setup
groq_api_key = st.secrets["GROQ_API_KEY"]
url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {groq_api_key}",
    "Content-Type": "application/json"
}
payload = {
    "model": "llama3-70b-8192",
    "messages": [
        {"role": "system", "content": "You are a professional DeFi protocol analyst."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.5,
    "top_p": 0.9,
    "max_tokens": 600
}

col1, col2 = st.columns([1.1, 1])

with col1:
    st.markdown("📊 Last 30 Days Data")
    st.dataframe(context_data)

with col2:
    st.markdown("#### 🎭 AI Scenario Insight")

    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        if "choices" in result and result["choices"]:
            output = result['choices'][0]['message']['content']
            bullets = [
                f"<li>{line.strip().lstrip('-• ')}</li>"
                for line in output.strip().split("\n")
                if line.strip()
            ]
            bullet_html = "<ul style='padding-left: 18px; padding-right: 8px;'>" + "\n".join(bullets) + "</ul>"
            st.markdown(bullet_html, unsafe_allow_html=True)
        else:
            st.warning("💭 Wait ... I'm thinking! 🤔")

    except Exception as e:
        st.error("AI Insight error.")
        st.caption(f"Exception: {e}")
