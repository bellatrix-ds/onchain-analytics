
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


# __________________ Import Data ______________________________________________________________________

df = pd.read_csv('https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/02_Uniswap_LP_Analysis/01_uniswap_daily_events_count.csv', on_bad_lines='skip')

st.set_page_config(layout="wide")
st.title("Uniswap Onchain LP Activity & Yield Monitor")



df['date'] = pd.to_datetime(df['date'] , errors="coerce")

fig = make_subplots(rows=1, cols=2, subplot_titles=("Mint vs Burn", "Swap Volume"))

fig.add_trace(go.Scatter(x=df['date'], y=df['Mint'], mode='lines+markers', name='Mint', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=df['date'], y=df['Burn'], mode='lines+markers', name='Burn', line=dict(color='red')), row=1, col=1)

fig.add_trace(go.Scatter(x=df['date'], y=df['Swap'], mode='lines+markers', name='Swap', line=dict(color='orange')), row=1, col=2)

fig.update_layout(
    title_text='Uniswap LP Daily Events (Onchain)',
    height=500,
    width=1000,
    showlegend=True
)

fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=2)

fig.show()
