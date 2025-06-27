
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

df = pd.read_csv('https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/02_Uniswap_LP_Analysis/01_uniswap_daily_events_count.csv', on_bad_lines='skip')

st.set_page_config(layout="wide")
st.title("Uniswap Onchain LP Activity & Yield Monitor")
