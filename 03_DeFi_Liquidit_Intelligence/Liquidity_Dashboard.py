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

filtered_data['month'] = filtered_data['block_timestamp'].dt.month
month_map = {3: 'March', 4: 'April', 5: 'May', 6: 'June'}
filtered_data['month_name'] = filtered_data['month'].map(month_map)
selected_month = st.radio("Select Month", options=['March', 'April', 'May', 'June'], horizontal=True)
month_num = [k for k, v in month_map.items() if v == selected_month][0]
filtered_month_data = filtered_data[filtered_data['month'] == month_num]

left_col, right_col = st.columns([1.5, 1])

with left_col:
    st.subheader("Utilization Rate Over Time")
    fig = px.line(
        filtered_month_data,
        x='block_timestamp',
        y='utilization_rate',
        markers=True,
        labels={'block_timestamp': 'Date', 'utilization_rate': 'Utilization Rate'}
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("ℹ️ توضیح")
    st.markdown("""
    این نمودار میزان استفاده از نقدینگی را در طول زمان برای استخر انتخاب‌شده نشان می‌دهد.  
    **بالا** می‌تواند نشانه فشار نقدینگی باشد. در حالی که نرخ **پایین** ممکن است به معنی سرمایه‌های بلااستفاده باشد.  
    نقاط **صفر** یا **نال** معمولاً به معنی فقدان فعالیت یا ثبت‌نشدن داده در آن روز خاص هستند.
    """)

# __________________ Part 1: Trends ______________________________________________________________________


# __________________ Part 2: Net Flow ______________________________________________________________________

col6, col7 = st.columns(2)

with col6:
    df_netflow = filtered_df[['block_timestamp', 'net_flow']].dropna()
    df_netflow['block_timestamp'] = pd.to_datetime(df_netflow['block_timestamp'])

    fig2 = px.area(df_netflow, x='block_timestamp', y='net_flow',
                   title='Net Flow Over Time', color_discrete_sequence=['#2196F3'])
    fig2.update_layout(xaxis_title='Date', yaxis_title='Net Flow', height=400)
    st.plotly_chart(fig2, use_container_width=True)

with col7:
    st.markdown("### 🔄 Net Flow یعنی چه؟")
    st.write("""
    **Net Flow = Deposit - Withdraw**  
    - مثبت: ورود نقدینگی بیشتر از برداشت  
    - منفی: خروج سرمایه بیشتر از ورودی  
    شناسایی Net Flow منفی به مدت چند روز پیاپی، ممکن است هشدار بحران نقدینگی باشد.
    """)
