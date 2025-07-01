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

import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px

# 🔧 Load the filtered Net Flow data
df_netflow = filtered_data[['block_timestamp', 'net_flow']].dropna()
df_netflow['block_timestamp'] = pd.to_datetime(df_netflow['block_timestamp'])

# 📊 Net Flow Chart (left column)
col1, col2 = st.columns([1, 1.1])
with col1:
    st.markdown("### 📈 Net Flow Over Time")
    fig = px.area(df_netflow, x='block_timestamp', y='net_flow',
                  color_discrete_sequence=['#2196F3'], height=400)
    fig.update_layout(xaxis_title='Date', yaxis_title='Net Flow')
    st.plotly_chart(fig, use_container_width=True)

# 🧠 AI Analysis Prompt Selector (right column)
with col2:
    st.markdown("### 🤖 AI-Powered Summary")
    analysis_type = st.selectbox("Select the type of analysis you want:", [
        "🔍 Key insights (3 bullets)",
        "⚠️ Risks and outflow alerts",
        "📈 Trend & volatility explanation",
        "🧠 Full narrative analysis"
    ])

    # Context: convert time series to plain text summary
    def make_context(df):
        context = ""
        for _, row in df.sort_values('block_timestamp').iterrows():
            date = row['block_timestamp'].strftime('%Y-%m-%d')
            context += f"{date}: {row['net_flow']:.2f}\n"
        return context.strip()

    context_text = make_context(df_netflow)

    # Prompt based on analysis type
    prompt_map = {
        "🔍 Key insights (3 bullets)": "Give exactly 3 non-obvious insights from the Net Flow time series. Do not explain what net flow is.",
        "⚠️ Risks and outflow alerts": "Analyze the net flow data and highlight any potential risks, large withdrawals, or warning signs in under 5 bullet points.",
        "📈 Trend & volatility explanation": "Explain the trends and volatility of the Net Flow values over time. Be concise and skip basic definitions.",
        "🧠 Full narrative analysis": "Provide a full analytical narrative of the Net Flow behavior in this DeFi lending pool. Include major changes and possible causes."
    }
    prompt = prompt_map[analysis_type]

    # 🔑 Together API config
    TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
    TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]

    def ask_ai(prompt, context):
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "Qwen/QwQ-32B",
            "messages": [
                {"role": "system", "content": "You are a DeFi data analyst."},
                {"role": "user", "content": f"Context:\n{context}"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1024
        }
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
        if response.ok:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"❌ API Error: {response.status_code}, {response.text}")

    # 🔍 Run AI Analysis
    with st.spinner("Analyzing Net Flow using AI..."):
        try:
            output = ask_ai(prompt, context_text)
            st.markdown("### ✅ AI Analysis")
            for line in output.strip().split("\n"):
                if line.strip().startswith("-") or line.strip().startswith("•"):
                    st.markdown(f"{line.strip()}")
        except Exception as e:
            st.error(f"❌ AI Analysis failed: {e}")


st.markdown("___")
