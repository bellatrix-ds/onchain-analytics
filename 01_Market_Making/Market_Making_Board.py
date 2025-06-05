#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --------------------------------------
# Load main dataset
df = pd.read_csv(
    'https://raw.githubusercontent.com/bellatrix-ds/ml-in-crypto/refs/heads/main/03_Wallet_Identity_Classifier/04_df_final.csv',
    on_bad_lines='skip'
)

# --------------------------------------
# Streamlit UI
st.title("🧠 Onchain Wallet Profiler")

category_emojis = {
    "Dex Trader": "📈",
    "Protocol Dev": "🛠️",
    "Yield Farmer": "🌾",
    "Nft Collector": "🔼️",
    "Oracle User": "🔮",
    "Staker Validator": "🗳️",
    "Defi Farmer": "👨‍🌾",
    "Bot": "🤖",
    "Bridge User": "🌉",
    "Airdrop Hunter": "🎯"
}

# Wallet selector
selected_wallet = st.selectbox("🔍 Select a wallet address", df["FROM_ADDRESS"].unique())

# Category for selected wallet
selected_category = df[df["FROM_ADDRESS"] == selected_wallet]["TOP_PROFILE"].values[0]
emoji = category_emojis.get(selected_category, "❓")

st.markdown(f"### 🏷️ Wallet Classified As: **{emoji} {selected_category}**")

# --------------------------------------
# Simulated market share
category_distribution = {
    "Dex Trader": 30,
    "Protocol Dev": 2,
    "Yield Farmer": 10,
    "Nft Collector": 15,
    "Oracle User": 5,
    "Staker Validator": 9,
    "Defi Farmer": 10,
    "Bot": 2,
    "Bridge User": 12,
    "Airdrop Hunter": 10
}
market_share = category_distribution.get(selected_category, 0)
st.markdown(f"### 🧩 This Category Powers **{market_share}%** of the Chain")
st.markdown("---")

#‌ـــــــ

metrics_df = pd.read_csv(
    'https://raw.githubusercontent.com/bellatrix-ds/ml-in-crypto/refs/heads/main/03_Wallet_Identity_Classifier/metrics_df.csv',
    on_bad_lines='skip')

# st.subheader("📌 Category Card")
st.markdown(f"### 📌 **{selected_category}** Category by the Numbers")


selected_metrics = metrics_df[metrics_df['TOP_PROFILE'] == selected_category].squeeze()
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("📊 Avg Tx per Month", f"{selected_metrics['avg_tx_per_month']:.0f}")
col2.metric("🔣 Unique Function Count", f"{selected_metrics['unique_function_count']}")
col3.metric("📜 Unique Contract Count", f"{selected_metrics['unique_contract_count']}")
col4.metric("⛽ Avg Gas Used", f"{selected_metrics['avg_gas_used']:.0f}")

#st.subheader("📊 Category Metrics Comparison")
st.markdown(f"### 🤺 **{selected_category}** vs. Other Onchain Beasts")


highlight_color = "#2CA02C"  
default_color = "#DDDDDD"  

metrics_df["COLOR"] = metrics_df["TOP_PROFILE"].apply(
    lambda x: highlight_color if x == selected_category else default_color
)

# ----------------------
def create_bar_chart(y_col, title):
    fig = px.bar(
        metrics_df,
        x="TOP_PROFILE",
        y=y_col,
        color="COLOR",
        color_discrete_map="identity",  
        title=title
    )
    fig.update_layout(showlegend=False)
    return fig

# ----------------------
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(create_bar_chart("avg_tx_per_month", "Avg Tx per Month"), use_container_width=True)
with col2:
    st.plotly_chart(create_bar_chart("unique_function_count", "Unique Function Count"), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(create_bar_chart("unique_contract_count", "Unique Contract Count"), use_container_width=True)
with col4:
    st.plotly_chart(create_bar_chart("avg_gas_used", "Avg Gas Used"), use_container_width=True)

st.markdown("---")


# Line chart data preparation
# Load trace dataset (df2) - already monthly
df2 = pd.read_csv(
    'https://raw.githubusercontent.com/bellatrix-ds/ml-in-crypto/refs/heads/main/03_Wallet_Identity_Classifier/line_chart.csv',
    on_bad_lines='skip')


df2['MONTH'] = pd.to_datetime(df2['MONTH'], errors='coerce')

# 👥 Merge TOP_PROFILE into df2 if not already there
if 'TOP_PROFILE' not in df2.columns:
    df2 = df2.merge(df[['FROM_ADDRESS', 'TOP_PROFILE']].drop_duplicates(), on='FROM_ADDRESS', how='left')

# 🧠 Filter data for selected wallet & its category
wallet_df = df2[df2["FROM_ADDRESS"] == selected_wallet]
category_df = df2[df2["TOP_PROFILE"] == selected_category]

# 🧮 Group and rename with correct column names
wallet_counts = wallet_df.groupby("MONTH")["WALLET_TX_COUNT"].sum().reset_index()
category_counts = category_df.groupby("MONTH")["CATEGORY_TX_MEAN"].mean().reset_index()

# 🔄 Merge and clean
merged = pd.merge(wallet_counts, category_counts, on="MONTH", how="outer").fillna(0)
merged = merged.sort_values("MONTH")
merged["MONTH_LABEL"] = merged["MONTH"].dt.strftime('%b-%Y')  # e.g. Jan-2024

# --------------------------------------
# 📈 Line chart
st.markdown("---")
#st.markdown(f"### 🤺 **{selected_category}** vs. Other Onchain Beasts")


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=merged["MONTH_LABEL"],
    y=merged["WALLET_TX_COUNT"],
    mode="lines+markers",
    name="Select Wallet Transactions",
    line=dict(color="blue", width=2)
))

fig.add_trace(go.Scatter(
    x=merged["MONTH_LABEL"],
    y=merged["CATEGORY_TX_MEAN"],
    mode="lines+markers",
    name=f"{selected_category} Category",
    line=dict(color="orange", width=2, dash="dash")
))

fig.update_layout(
    title=(f" 👛 This Wallet’s Activity vs. {selected_category} Category"),
    xaxis_title="Month",
    yaxis_title="Transaction Count",
    legend_title="Legend",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
# ----------------- GAS USED -----------------
# 📥 Load preprocessed gas usage data
df_gas = pd.read_csv(
    'https://raw.githubusercontent.com/bellatrix-ds/ml-in-crypto/refs/heads/main/03_Wallet_Identity_Classifier/line_chart_gas.csv',
    on_bad_lines='skip'
)

# 🧠 Filter data for selected wallet & category
wallet_gas = df_gas[df_gas["FROM_ADDRESS"] == selected_wallet]
category_gas = df_gas[df_gas["TOP_PROFILE"] == selected_category]

# 📊 Group monthly sums and means
wallet_gas_grouped = wallet_gas.groupby("MONTH")["WALLET_GAS_USED"].sum().reset_index()
category_gas_grouped = category_gas.groupby("MONTH")["CATEGORY_GAS_MEAN"].mean().reset_index()

# 🔁 Merge both time series
merged_gas = pd.merge(wallet_gas_grouped, category_gas_grouped, on="MONTH", how="outer").fillna(0)
merged_gas = merged_gas.sort_values("MONTH")

# 🛠️ Fix month format for plotting
merged_gas["MONTH"] = pd.to_datetime(merged_gas["MONTH"].astype(str), format="%Y-%m", errors="coerce")
merged_gas["MONTH_LABEL"] = merged_gas["MONTH"].dt.strftime('%b-%Y')  # e.g. Jan-2024

# 📈 Line chart for Gas Used
st.markdown("---")
fig_gas = go.Figure()

fig_gas.add_trace(go.Scatter(
    x=merged_gas["MONTH_LABEL"],
    y=merged_gas["WALLET_GAS_USED"],
    mode="lines+markers",
    name="Wallet Gas Used",
    line=dict(color="blue", width=2)
))

fig_gas.add_trace(go.Scatter(
    x=merged_gas["MONTH_LABEL"],
    y=merged_gas["CATEGORY_GAS_MEAN"],
    mode="lines+markers",
    name=f"{selected_category} Category",
    line=dict(color="orange", width=2, dash="dash")
))

fig_gas.update_layout(
    title= (f" ⛽ Wallet Gas Used vs. {selected_category} Category"),
    xaxis_title="Month",
    yaxis_title="Gas Used",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig_gas, use_container_width=True)


# ----------------- Value -----------------

df_value = pd.read_csv(
    'https://raw.githubusercontent.com/bellatrix-ds/ml-in-crypto/refs/heads/main/03_Wallet_Identity_Classifier/line_chart_value.csv',
    on_bad_lines='skip'
)

# 🧠 Filter data for selected wallet & category
wallet_value = df_value[df_value["FROM_ADDRESS"] == selected_wallet]
category_value = df_value[df_value["TOP_PROFILE"] == selected_category]

# 📊 Group monthly sums and means
wallet_value_grouped = wallet_value.groupby("MONTH")["WALLET_TOTAL_VALUE"].sum().reset_index()
category_value_grouped = category_value.groupby("MONTH")["CATEGORY_VALUE_MEAN"].mean().reset_index()

# 🔁 Merge both time series
merged_value = pd.merge(wallet_value_grouped, category_value_grouped, on="MONTH", how="outer").fillna(0)
merged_value = merged_value.sort_values("MONTH")

# 🛠️ Fix month format
merged_value["MONTH"] = pd.to_datetime(merged_value["MONTH"].astype(str), format="%Y-%m", errors="coerce")
merged_value["MONTH_LABEL"] = merged_value["MONTH"].dt.strftime('%b-%Y')

# 🧮 Convert value from Wei to ETH
merged_value["WALLET_TOTAL_VALUE"] = merged_value["WALLET_TOTAL_VALUE"] / 1e18
merged_value["CATEGORY_VALUE_MEAN"] = merged_value["CATEGORY_VALUE_MEAN"] / 1e18

# --------------------------------------
# 📈 Line chart for Transfer Value
st.markdown("---")
fig_value = go.Figure()

fig_value.add_trace(go.Scatter(
    x=merged_value["MONTH_LABEL"],
    y=merged_value["WALLET_TOTAL_VALUE"],
    mode="lines+markers",
    name="Wallet Transfer Value (ETH)",
    line=dict(color="blue", width=2)
))

fig_value.add_trace(go.Scatter(
    x=merged_value["MONTH_LABEL"],
    y=merged_value["CATEGORY_VALUE_MEAN"],
    mode="lines+markers",
    name=f"{selected_category} Category (Avg)",
    line=dict(color="orange", width=2, dash="dash")
))

fig_value.update_layout(
    title= (f" 💸 Wallet Transfer Value vs. {selected_category} Category"),
    xaxis_title="Month",
    yaxis_title="Transfer Value (ETH)",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig_value, use_container_width=True)

# ـــــــــ
weekday_activity = df = pd.read_csv('https://raw.githubusercontent.com/bellatrix-ds/ml-in-crypto/refs/heads/main/03_Wallet_Identity_Classifier/weekday_activity.csv', on_bad_lines='skip')
hourly_activity = pd.read_csv('https://raw.githubusercontent.com/bellatrix-ds/ml-in-crypto/refs/heads/main/03_Wallet_Identity_Classifier/hourly_activity.csv',on_bad_lines='skip')
# Clean and ensure correct order
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_activity['WEEKDAY'] = pd.Categorical(weekday_activity['WEEKDAY'], categories=weekday_order, ordered=True)

# Filter data for selected category
filtered_weekday = weekday_activity[weekday_activity['TOP_PROFILE'] == selected_category]
filtered_hourly = hourly_activity[hourly_activity['TOP_PROFILE'] == selected_category]

# Pivot for heatmaps
pivot_weekday = filtered_weekday.pivot(index='TOP_PROFILE', columns='WEEKDAY', values='UNIQUE_WALLETS')
pivot_hourly = filtered_hourly.pivot(index='TOP_PROFILE', columns='HOUR', values='UNIQUE_WALLETS')

# --------------------------------------
# Plot heatmaps
st.subheader("📅 Weekly Activity Pattern")
fig1, ax1 = plt.subplots(figsize=(10, 1.5))
sns.heatmap(pivot_weekday, annot=False, cmap="YlGnBu", cbar=True, ax=ax1)
ax1.set_ylabel("")
ax1.set_xlabel("")
st.pyplot(fig1)

st.subheader("⏰ Hourly Activity Pattern")
fig2, ax2 = plt.subplots(figsize=(10, 1.5))
sns.heatmap(pivot_hourly, annot=False, cmap="YlGnBu", cbar=True, ax=ax2)
ax2.set_ylabel("")
ax2.set_xlabel("")
st.pyplot(fig2)


# ـــ

st.markdown("---")
st.caption("Contact me: bellabahramii@gmail.com")


