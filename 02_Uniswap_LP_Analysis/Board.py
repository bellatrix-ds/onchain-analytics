
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
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# __________________ Import Data ______________________________________________________________________

df = pd.read_csv('https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/02_Uniswap_LP_Analysis/01_uniswap_daily_events_count.csv', on_bad_lines='skip')

st.set_page_config(layout="wide")
st.title("Uniswap Onchain LP Activity & Yield Monitor")



df['date'] = pd.to_datetime(df['date'] , errors="coerce")
col1, col2 = st.columns(2)

# چپ: Mint و Burn
with col1:
    st.subheader("Mint & Burn Events")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['date'], y=df['Mint'], mode='lines+markers', name='Mint', line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=df['date'], y=df['Burn'], mode='lines+markers', name='Burn', line=dict(color='red')))
    fig1.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig1, use_container_width=True)

# راست: Swap
with col2:
    st.subheader("Swap Events")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['date'], y=df['Swap'], mode='lines+markers', name='Swap', line=dict(color='orange')))
    fig2.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)
