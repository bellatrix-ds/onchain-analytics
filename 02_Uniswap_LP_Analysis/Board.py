
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

# st.set_page_config(layout="wide")
st.title("Uniswap Onchain LP Activity & Yield Monitor")



df['date'] = pd.to_datetime(df['date'] , errors="coerce")


st.subheader("Mint & Burn Events")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df['date'], y=df['Mint'], mode='lines', name='Mint', line=dict(color='green')))
fig3.add_trace(go.Scatter(x=df['date'], y=df['Burn'], mode='lines', name='Burn', line=dict(color='red')))
fig3.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig3, use_container_width=True)





df['Net_Mint'] = df['Mint'] - df['Burn']
df['Mint_smooth'] = df['Mint'].rolling(7).mean()
df['Burn_smooth'] = df['Burn'].rolling(7).mean()
# ──────────────────────
# بخش اول: نمودار Net Mint
# ──────────────────────

st.subheader("📊 Net Mint (Mint - Burn)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['Net_Mint'], mode='lines', name='Net Mint', line=dict(color='blue')))
fig.add_hline(y=0, line=dict(color='gray', dash='dash'))
fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)



    
