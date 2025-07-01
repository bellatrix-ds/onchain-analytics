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

import requests
import json



import streamlit as st
import pandas as pd
import plotly.express as px
import requests

API_KEY = "79e7ccd7e568ae5694594efe5e318a03fc42a64d4c7bc2dc491a6e2123404fd9"

MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

col6, col7 = st.columns(2)

with col6:
    df_netflow = filtered_data[['block_timestamp', 'net_flow']].dropna()
    df_netflow['block_timestamp'] = pd.to_datetime(df_netflow['block_timestamp'], errors='coerce')
    df_netflow = df_netflow.dropna().sort_values(by="block_timestamp")

    fig2 = px.area(df_netflow, x='block_timestamp', y='net_flow',
                   title='Net Flow Over Time', color_discrete_sequence=['#2196F3'])
    fig2.update_layout(xaxis_title='Date', yaxis_title='Net Flow', height=400)
    st.plotly_chart(fig2, use_container_width=True)

# خلاصه‌ساز امن برای AI
def make_summary_netflow(data: pd.DataFrame, limit=15) -> str:
    data = data.dropna(subset=['block_timestamp', 'net_flow']).copy()
    data['block_timestamp'] = pd.to_datetime(data['block_timestamp'], errors='coerce')
    data = data.dropna(subset=['block_timestamp']).sort_values(by="block_timestamp", ascending=False).head(limit)

    summary = ""
    for _, row in data.iterrows():
        summary += f"Date: {row['block_timestamp'].date()}, Net Flow: {row['net_flow']:.2f}\n"
    return summary.strip()

# درخواست از مدل Together
def ask_together_api(context: str) -> str:
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    context_clean = context.encode('ascii', errors='ignore').decode()

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a DeFi analyst. Provide an analytical summary based on Net Flow trends in a lending protocol."},
            {"role": "user", "content": f"Here is the Net Flow data:\n{context_clean}"},
            {"role": "user", "content": "Give a short summary of recent liquidity behavior and possible risks based on this data."}
        ],
        "temperature": 0.4,
        "top_p": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.ok:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"❌ Error: Status {response.status_code}, Body: {response.text}")

# گرفتن تحلیل و نمایش
with col7:
    st.markdown("### 🤖 تحلیل هوشمند Net Flow")
    try:
        context = make_summary_netflow(df_netflow)
        with st.spinner("در حال تحلیل نمودار توسط AI..."):
            explanation = ask_together_api(context)
            st.write(explanation)
    except Exception as e:
        st.error(f"❌ خطا در تحلیل AI: {str(e)}")

st.markdown("___")
