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
    selected_protocol = st.selectbox("Lending Protocol", sorted(data['lending_protocol'].dropna().unique()))
with col2:
    chains = data[data['lending_protocol'] == selected_protocol]['chain'].dropna().unique()
    selected_chain = st.selectbox("Chain", sorted(chains))
with col3:
    pools = data[
        (data['lending_protocol'] == selected_protocol) &
        (data['chain'] == selected_chain)
    ]['pool'].dropna().unique()
    selected_pool = st.selectbox("Pool", sorted(pools))

filtered_df = data[
    (data['lending_protocol'] == selected_protocol) &
    (data['chain'] == selected_chain) &
    (data['pool'] == selected_pool)]


# __________________ Part 1: Trends ______________________________________________________________________
import calendar

filtered_df['block_timestamp'] = pd.to_datetime(filtered_df['block_timestamp'])
filtered_df['month'] = filtered_df['block_timestamp'].dt.month
filtered_df['month_str'] = filtered_df['block_timestamp'].dt.month.apply(lambda x: calendar.month_name[x])

months_available = filtered_df['month_str'].unique()
selected_month = st.selectbox("Select Month", sorted(months_available))

df_plot = filtered_df[filtered_df['month_str'] == selected_month][['block_timestamp', 'utilization_rate']].dropna()

fig1 = px.line(df_plot, x='block_timestamp', y='utilization_rate',
               title='Utilization Rate Over Time', markers=True)
fig1.update_layout(xaxis_title='Date', yaxis_title='Utilization Rate', height=400)

st.plotly_chart(fig1, use_container_width=True)


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
