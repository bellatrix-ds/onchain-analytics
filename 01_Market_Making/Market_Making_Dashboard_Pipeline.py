#!/usr/bin/env python
# coding: utf-8

# In[18]:


# 📊 Data Handling
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# 📈 Visualization
import matplotlib.pyplot as plt
import plotly.express as px

# ⚙️ Preprocessing & Scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# In[ ]:


# Detect a complete list of all stable pools in selected dexs

API_KEY = "ZaxSVZrcZe5oxJmZV0djA1FyTGEOFFmG" 
QUERY_ID = 5282034   

def run_dune_query_by_id(query_id):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    submit = requests.post(
        f"https://api.dune.com/api/v1/query/{query_id}/execute",
        headers=headers
    )
    execution_id = submit.json().get("execution_id")
    if not execution_id:
        raise RuntimeError("Query execution failed to start.")

    while True:
        res = requests.get(f"https://api.dune.com/api/v1/execution/{execution_id}/results", headers=headers)
        status = res.json().get("state")
        if status == "QUERY_STATE_COMPLETED":
            break
        elif status in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            raise RuntimeError(f"Query failed with state {status}")
        time.sleep(3)

    data = res.json()["result"]["rows"]
    return pd.DataFrame(data)

df_all_pools = run_dune_query_by_id(QUERY_ID)


# In[ ]:


# Make a datafram for daily metrics based on filtered pools

API_KEY = "ZaxSVZrcZe5oxJmZV0djA1FyTGEOFFmG" 
QUERY_ID = 5273508   

def run_dune_query_by_id(query_id):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    submit = requests.post(
        f"https://api.dune.com/api/v1/query/{query_id}/execute",
        headers=headers
    )
    execution_id = submit.json().get("execution_id")
    if not execution_id:
        raise RuntimeError("Query execution failed to start.")

    while True:
        res = requests.get(f"https://api.dune.com/api/v1/execution/{execution_id}/results", headers=headers)
        status = res.json().get("state")
        if status == "QUERY_STATE_COMPLETED":
            break
        elif status in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            raise RuntimeError(f"Query failed with state {status}")
        time.sleep(3)

    data = res.json()["result"]["rows"]
    return pd.DataFrame(data)

df_metrics = run_dune_query_by_id(QUERY_ID)


# In[20]:


# data manupulation
#data = pd.read_csv('df_main.csv')
df_metrics = data.copy()
data['date'] = pd.to_datetime(data['date']).dt.date
data['Spread'] = pd.to_numeric(data['Spread'], errors='coerce') * 100
data['swap_count'] = pd.to_numeric(data['swap_count'], errors='coerce', downcast='integer')
data['volume'] = pd.to_numeric(data['volume'], errors='coerce')
data['order_size'] = pd.to_numeric(data['order_size'], errors='coerce')


# In[22]:


# create bins for order size

bins = [0, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 1_000_000, float('inf')]
labels = [
    '≤1k', '1k–5k', '5k–10k', '10k–25k', '25k–50k',
    '50k–100k', '100k–250k', '250k–1M', '>1M'
]

data['order_size_bin'] = pd.cut(data['order_size'], bins=bins, labels=labels)


# In[24]:


# create bins for volume

volume_bins = [
    0,
    1e5,        # 100k
    5e5,        # 500k
    1e6,        # 1M
    5e6,        # 5M
    10e6,       # 10M
    50e6,       # 50M
    100e6,      # 100M
    float('inf')
]

volume_labels = [
    '≤100k', '100k–500k', '500k–1M', '1M–5M', '5M–10M',
    '10M–50M', '50M–100M', '>100M'
]

data['volume_bin'] = pd.cut(data['volume'], bins=volume_bins, labels=volume_labels)


# In[26]:


# calculate trade size and bins for it

data['Trade_size'] = data['volume'] / data['swap_count']

bins = [
    0,
    10_000,
    50_000,
    100_000,
    200_000,
    300_000,
    400_000,
    500_000,
    750_000,
    1_000_000,
    np.inf
]

labels = [
    '≤10k', '10k–50k', '50k–100k', '100k–200k',
    '200k–300k', '300k–400k', '400k–500k',
    '500k–750k', '750k–1M', '>1M'
]

data['trade_size_bin'] = pd.cut(data['Trade_size'], bins=bins, labels=labels)


# In[28]:


# Ctreating Scoring System

pool_stats = data.groupby(["blockchain","dex","pool"]).agg({
    "volume": "mean",
    "swap_count": "mean",
    "order_size": "mean",
    "Spread": ["mean", "std"],
    "Trade_size": "mean"
})

pool_stats.columns = ['volume_mean', 'swap_count_mean', 'order_size_mean', 'spread_mean', 'spread_std', 'trade_size_mean']
pool_stats = pool_stats.reset_index()

scaler = MinMaxScaler()
metrics = ['volume_mean', 'swap_count_mean', 'order_size_mean', 'spread_mean', 'spread_std', 'trade_size_mean']

pool_stats_normalized = pool_stats.copy()
pool_stats_normalized[metrics] = scaler.fit_transform(pool_stats[metrics])

pool_stats_normalized["mm_score"] = (
    pool_stats_normalized["volume_mean"] * 0.25 +
    pool_stats_normalized["swap_count_mean"] * 0.2 +
    pool_stats_normalized["order_size_mean"] * 0.1 +
    (1 - pool_stats_normalized["spread_mean"]) * 0.25 +
    (1 - pool_stats_normalized["spread_std"]) * 0.1 +
    pool_stats_normalized["trade_size_mean"] * 0.1
)

top_pools = pool_stats_normalized.sort_values(by="mm_score", ascending=False)


# In[29]:


data


# In[ ]:




