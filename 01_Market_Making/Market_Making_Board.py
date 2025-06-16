#!/usr/bin/env python
# coding: utf-8

# In[ ]:
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

# __________________ Import Data ______________________________________________________________________

data = pd.read_csv('https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/01_Market_Making/df_main_1.csv', on_bad_lines='skip')

st.set_page_config(layout="wide")
st.title("🔍 Stable Pools Market Maker Radar")

def custom_metric(title, value, subtext=None, color="green"):
    html = f"""
        <div style='text-align: center; padding: 8px'>
            <div style='font-size:15px; color: gray'>{title}</div>
            <div style='font-size:20px; font-weight:600'>{value}</div>
    """
    if subtext:
        html += f"<div style='font-size:15px; color:{color}'>↑ {subtext}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# __________________ Part1: Key KPIs ______________________________________________________________________

data["date"] = pd.to_datetime(data["date"], errors="coerce")

# Define rolling time windows
latest_7d = data[data["date"] >= data["date"].max() - pd.Timedelta(days=7)]
latest_3d = data[data["date"] >= data["date"].max() - pd.Timedelta(days=3)]
latest_14d = data[data["date"] >= data["date"].max() - pd.Timedelta(days=14)]

# Top MM score pool
latest_7d["mm_score"] = (
    latest_7d["volume"].rank(pct=True) * 0.3 +
    latest_7d["swap_count"].rank(pct=True) * 0.2 +
    latest_7d["Trade_size"].rank(pct=True) * 0.2 +
    (1 - latest_7d["Spread"].rank(pct=True)) * 0.3
)
top_mm_row = latest_7d.loc[latest_7d["mm_score"].idxmax()]
top_pool_name = top_mm_row["pool"]
top_pool_score = f"Score: {top_mm_row['mm_score']:.3f}"

# Highest volume pool
top_volume_row = latest_7d.loc[latest_7d["volume"].idxmax()]
top_volume_pool = top_volume_row["pool"]
top_volume_val = f"${top_volume_row['volume']:,.0f}"

# Most volatile pool
spread_vol = latest_3d.groupby("pool")["Spread"].std()
if not spread_vol.empty:
    most_volatile_pool = spread_vol.idxmax()
    most_volatile_val = f"Spread Std: {spread_vol.max():.4f}"
else:
    most_volatile_pool, most_volatile_val = "N/A", None

# Risky pool with most occurrences of Spread > 5%
risky_pool_series = latest_7d[latest_7d["Spread"] > 0.05]["pool"]
if not risky_pool_series.empty:
    risky_pool_name = risky_pool_series.value_counts().idxmax()
    risky_pool_val = f"Spread > 5% x {risky_pool_series.value_counts().max()}"
else:
    risky_pool_name, risky_pool_val = "N/A", None

# Most efficient DEX (lowest median spread)
dex_median_spread = latest_14d.groupby("dex")["Spread"].median()
if not dex_median_spread.empty:
    best_dex = dex_median_spread.idxmin()
    best_dex_val = f"Median Spread: {dex_median_spread.min():.4%}"
else:
    best_dex, best_dex_val = "N/A", None

# Final display
st.markdown("#### 📊 Part 1: Key Market-Making Insights (last 7 days)")
st.markdown(" 👀 Let’s take a quick glance at what stood out over the past 7 days: top pools, wild spreads, and where the action’s been.")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    custom_metric("🏆 Top Pool by MM Score", top_pool_name, top_pool_score)
with col2:
    custom_metric("💸 Highest Volume Pool", top_volume_pool, top_volume_val)
with col3:
    custom_metric("⚠️ Most Volatile Pool", most_volatile_pool, most_volatile_val)
with col4:
    custom_metric("🚨 Risky Pool", risky_pool_name, risky_pool_val)
with col5:
    custom_metric("📉 Most Efficient DEX", best_dex, best_dex_val)
st.markdown(" ")
st.markdown(" ")
st.markdown("---")
st.markdown(" ")

# __________________ Part2: Opportunity Scanner______________________________________________________________________

st.markdown("### 💡 Part 2: Opportunity Scanner")
st.markdown("  🧭 Find pools where the big guys are missing, low competition, juicy spreads, and solid trade volume.")

# __________________ 2.1: ⚡ Recent Spike Alerts______________________________________________________________________

df = data.copy()

df['date'] = pd.to_datetime(df['date'])

df_sorted = df.sort_values(by=['pool', 'date'])
df_sorted['volume_2d_change'] = df_sorted.groupby('pool')['volume'].pct_change(periods=2)

# --- Filter for spike conditions ---
spike_df = df_sorted[
    (df_sorted['volume_2d_change'] > 3.0) & 
    (df_sorted['Spread'] < 0.01)            
].copy()

# --- Pick top 6 spikes from different pools ---
unique_pools = []
final_spikes = []

for _, row in spike_df.sort_values(by='volume_2d_change', ascending=False).reset_index().iterrows():
    if row['pool'] not in unique_pools:
        final_spikes.append(row)
        unique_pools.append(row['pool'])
    if len(final_spikes) == 6:
        break

# --- Format for display ---
spike_summary_df = pd.DataFrame(final_spikes)[["date", "pool", "volume", "volume_2d_change", "Spread"]].copy()
spike_summary_df["Date"] = spike_summary_df["date"].dt.strftime("%B %d")
spike_summary_df["Volume"] = spike_summary_df["volume"].apply(lambda x: f"${x/1_000_000:.1f}M")
spike_summary_df["% Change"] = spike_summary_df["volume_2d_change"].apply(lambda x: f"+{int(x*100):,}%")
spike_summary_df["Spread"] = spike_summary_df["Spread"].apply(lambda x: f"{x*100:.2f}%")
spike_summary_df["🔥 Note"] = "Pool just woke up"
display_table = spike_summary_df[["Date", "pool", "Volume", "% Change", "Spread", "🔥 Note"]]

st.markdown(" ")

# --- Streamlit layout ---
col_left, col_right = st.columns([1, 1])

# --- Left column: compact insights ---
with col_left:
    st.markdown("##### ⚡ 2.1: Recent Spike Alerts")
    for _, row in spike_summary_df.iterrows():
        st.markdown(
            f"📉 **{row['Date']}** : `{row['pool']}` traded at **{row['Volume']}** with a "
            f"**{row['% Change']} jump** (spread: **{row['Spread']}**) 🔥"
        )

# --- Right column: summary table ---
with col_right:
    st.dataframe(display_table, use_container_width=True)


st.markdown(" ")
st.markdown(" ")
st.markdown(" ")
st.markdown(" ")


# __________________ 2.2: Low-Competition Pools______________________________________________________________________

df = data.copy()

grouped = df.groupby(["pool", "pool_id"]).agg({
    "volume": "sum",
    "Spread": "mean",          
    "order_size": "mean",
    "swap_count": "mean",
    "date": "count"
}).rename(columns={"Spread": "Spread_raw", "date": "num_days"})

grouped["liquidity_est"] = grouped["order_size"] * 2

grouped["estimated_fee_total"] = grouped["volume"] * grouped["Spread_raw"]
grouped["fee_per_day"] = grouped["estimated_fee_total"] / grouped["num_days"]
grouped["APR"] = (grouped["fee_per_day"] / grouped["liquidity_est"]) * 365

grouped = grouped.reset_index()
grouped["APR (%)"] = (grouped["APR"] ).round(2)
grouped["Spread (%)"] = (grouped["Spread_raw"] ).round(2)
grouped["Swap Count"] = grouped["swap_count"].round(0).astype(int)
grouped["Volume ($)"] = (grouped["volume"] / 1_000_000).round(1).astype(str) + "M"

# --- Apply Low-Competition Filter
filtered = grouped[
    (grouped["APR (%)"] > 10) &
    (grouped["Spread (%)"] > 2) &
    (grouped["Swap Count"] < 100)
]

top5 = filtered.sort_values("APR (%)", ascending=False).head(5)
display_df = top5[["pool", "APR (%)", "Spread (%)", "Swap Count", "Volume ($)"]]

# --- Two-column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("##### 💡 2.2: Low-Competition Pools")
    st.markdown("""
These are the **Top 5 Pools** where:

- ✅ APR is high → lots of fees to capture  
- ❌ Spread is wide → inefficient pricing  
- ❌ Swap activity is low → weak or no market makers  

Perfect opportunities for early MM entry before competition intensifies.
""")

with col2:
    st.markdown(" ")
    st.dataframe(
        display_df.style.format({
            "APR (%)": "{:.2f}",
            "Spread (%)": "{:.2f}",
            "Swap Count": "{:,.0f}"
        }),
        use_container_width=True
    )


st.markdown(" ")
st.markdown("---")
st.markdown(" ")

# __________________ Build Filters ______________________________________________________________________

st.markdown("### 🌐 Part 3: The Big Picture, All Markets")
st.markdown("  📍 Big picture time. Want to zoom into a chain, DEX, or pool? Go ahead, filters are yours.")

# __________________ Build Filters ______________________________________________________________________

st.markdown(" ")


trade_size_order = [
    "All", "≤10k", "10k–50k", "50k–100k", "100k–200k",
    "200k–300k", "300k–400k", "400k–500k",
    "500k–750k", "750k–1M", ">1M"
]

col1, col2, col3, col4 = st.columns(4)
selected_chain = col1.selectbox("Select Blockchain", ["All"] + sorted(data["blockchain"].unique().tolist()))
selected_dex = col2.selectbox("Select DEX", ["All"] + sorted(data["dex"].unique().tolist()))

if selected_dex == "All":
    filtered_pool_options = data["pool"].unique().tolist()
else:
    filtered_pool_options = data[data["dex"] == selected_dex]["pool"].unique().tolist()

selected_pool = col3.selectbox("Select Pool", ["All"] + sorted(filtered_pool_options))
min_trade_size = col4.selectbox("Minimum Trade Size ($)", options=trade_size_order, index=0)

filtered_data = data.copy()
if selected_chain != "All":
    filtered_data = filtered_data[filtered_data["blockchain"] == selected_chain]

if selected_dex != "All":
    filtered_data = filtered_data[filtered_data["dex"] == selected_dex]

if selected_pool != "All":
    filtered_data = filtered_data[filtered_data["pool"] == selected_pool]


if min_trade_size != "All":
    trade_size_bins = trade_size_order[1:]
    selected_index = trade_size_bins.index(min_trade_size)
    allowed_bins = trade_size_bins[selected_index:]
    filtered_data = filtered_data[filtered_data["trade_size_bin"].isin(allowed_bins)]


# __________________ Part3.1: Pie chart + Pool Count  ______________________________________________________________________

st.markdown(" ")
st.markdown(" ")
st.markdown(" ")


st.markdown("##### 📈 3.1: Pie chart + Pool Count")

grouped = filtered_data.groupby("pool").agg({
    "swap_count": "sum",
    "Trade_size": "mean"
}).reset_index()

grouped.rename(columns={"pool": "pool_name"}, inplace=True)

grouped["swap_share"] = grouped["swap_count"] / grouped["swap_count"].sum()
grouped["trade_size_share"] = grouped["Trade_size"] / grouped["Trade_size"].sum()

fig1 = px.pie(
    grouped,
    values='swap_share',
    names='pool_name',
    title=f"📌 Market Share by Swap Count on {selected_dex} ({selected_chain})",
    hole=0.45
)

fig2 = px.pie(
    grouped,
    values='trade_size_share',
    names='pool_name',
    title=f"📌 Market Share by Trade Size on {selected_dex} ({selected_chain})",
    hole=0.45
)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.plotly_chart(fig2, use_container_width=True)



st.markdown("---")


# __________________ Part 3.2: Trade Size vs. Slippage ______________________________________________________________________

st.markdown(" ")
st.markdown(" ")
st.markdown(" ")


st.markdown("##### 📈 3.2: Spread vs. Trade Size")

filtered_data['Spread'] = filtered_data['Spread'] / 100

trade_size_order = [
    "≤10k", "10k–50k", "50k–100k", "100k–200k", "200k–300k", "300k–400k",
    "400k–500k", "500k–750k", "750k–1M", ">1M"
]

trade_size_options = ["All"] + trade_size_order

col_text1, col_chart1 = st.columns([1, 1])

with col_text1:
    st.markdown(f"""
    - 🔍 Here, we want to understand how the spread percentage changes in a pool relative to the trade size.
    - **Use case**: We want to know what trade size we can enter with, in order to avoid slippage risk while still capturing a good profit opportunity.
    - **Look for**: Pools with **steep spread increase** {f"after `{min_trade_size}`" if min_trade_size != "All" else "at any trade size"}.
    - These pools show **critical price slippage {f"after {min_trade_size} trades" if min_trade_size != "All" else "across ranges"}**.  
      If you can provide depth, you’ll dominate pricing.
    """)

def compute_slopes(filtered_data, allowed_bins):
    if allowed_bins == "All":
        return df["pool"].unique().tolist()
    
    threshold_idx = trade_size_order.index(allowed_bins)
    result = []
    for pool, group in df.groupby("pool"):
        group = group.copy()
        group["x_idx"] = group["trade_size_bin"].apply(lambda x: trade_size_order.index(x))
        group = group.sort_values("x_idx")
        post_threshold = group[group["x_idx"] >= threshold_idx]
        if len(post_threshold) >= 2:
            x = post_threshold["x_idx"]
            y = post_threshold["Spread"]
            slope = (y.iloc[-1] - y.iloc[0]) / (x.iloc[-1] - x.iloc[0])
            if slope > 0.01:
                result.append(pool)
    return result

steep_pools = compute_slopes(filtered_data, min_trade_size)
filtered_chart_data = filtered_data[filtered_data["pool"].isin(steep_pools)]

with col_chart1:
    chart = alt.Chart(filtered_chart_data).mark_line(point=True).encode(
        x=alt.X("trade_size_bin", title="Trade Size (binned)", sort=trade_size_order, axis=alt.Axis(labelAngle=30)),
        y=alt.Y("Spread", title="Spread (%)"),
        color="pool"
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)


st.markdown("---")
st.markdown(" ")








# __________________ Part3: Heatmap ______________________________________________________________________

st.subheader("🔥 Slippage Heatmap (Pool vs Order Size)")

order_size_order = [
    "≤1k", "1k–5k", "5k–10k", "10k–25k", "25k–50k",
    "50k–100k", "100k–250k", "250k–1M", ">1M"
]


col_text2  , col_chart2 = st.columns([1, 1])

with col_chart2:
    heatmap = alt.Chart(filtered_data).mark_rect().encode(
        x=alt.X("order_size_bin", title="Order Size (binned)",axis=alt.Axis(labelAngle=30, labelOverlap=False),sort=order_size_order),
        y=alt.Y("pool", title="Pool"),
        color=alt.Color("Spread:Q", scale=alt.Scale(scheme='redyellowgreen', reverse=True), title="Spread (%)")
    ).properties(height=500)

    st.altair_chart(heatmap, use_container_width=True)

with col_text2:
    st.markdown("### What to look for?")
    st.markdown("""
    - **Goal**: Spot liquidity weakness by order size and pool.
    - **Look for**: Brighter red blocks—higher slippage means riskier for MM.
    """)

st.markdown("---")

# __________________ Part4: Scoring System ______________________________________________________________________
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

st.subheader("🧠 Pool Scoring System")

col_left, col_right = st.columns([1, 1])

with col_left:
    preset = st.radio(
        "Choose your market making strategy profile:",
        options=[
            "🐢 Low-risk, deep pool hunter",
            "⚡ High-APR sniper",
            "🎯 Balance seeker (PnL max)"
        ],
        index=0
    )

if preset == "🐢 Low-risk, deep pool hunter":
    volume_w = 0.15
    swap_w = 0.10
    order_size_w = 0.15
    trade_size_w = 0.20
    spread_mean_w = 0.25
    spread_std_w = 0.15
elif preset == "⚡ High-APR sniper":
    volume_w = 0.30
    swap_w = 0.25
    order_size_w = 0.10
    trade_size_w = 0.15
    spread_mean_w = 0.10
    spread_std_w = 0.10
else:  # 🎯 Balance seeker (PnL max)
    volume_w = 0.25
    swap_w = 0.20
    order_size_w = 0.10
    trade_size_w = 0.15
    spread_mean_w = 0.20
    spread_std_w = 0.10

total_weight = volume_w + swap_w + order_size_w + spread_mean_w + spread_std_w + trade_size_w

# Make sure your DataFrame is called `data` and has the required columns
pool_stats = data.groupby(["blockchain", "dex", "pool"]).agg({
    "volume": "mean",
    "swap_count": "mean",
    "order_size": "mean",
    "Spread": ["mean", "std"],
    "Trade_size": "mean"
})
pool_stats.columns = ['volume_mean', 'swap_count_mean', 'order_size_mean', 'spread_mean', 'spread_std', 'trade_size_mean']
pool_stats = pool_stats.reset_index()

# ---- Normalize metrics ----
scaler = MinMaxScaler()
metrics = ['volume_mean', 'swap_count_mean', 'order_size_mean', 'spread_mean', 'spread_std', 'trade_size_mean']
pool_stats_normalized = pool_stats.copy()
pool_stats_normalized[metrics] = scaler.fit_transform(pool_stats[metrics])

# ---- Calculate MM Score ----
pool_stats_normalized["mm_score"] = (
    (pool_stats_normalized["volume_mean"] * volume_w) +
    (pool_stats_normalized["swap_count_mean"] * swap_w) +
    (pool_stats_normalized["order_size_mean"] * order_size_w) +
    ((1 - pool_stats_normalized["spread_mean"]) * spread_mean_w) +
    ((1 - pool_stats_normalized["spread_std"]) * spread_std_w) +
    (pool_stats_normalized["trade_size_mean"] * trade_size_w)
) / total_weight

# ---- Results and Layout ----
with col_left:
    st.markdown("###### ℹ️ How This Scoring Works")
    st.markdown(f"""
    This scoring system evaluates how suitable a pool is for **market making** based on the selected strategy preset:
    

    
    **Metrics considered:**
    - ✅ High volume, trade size, swap activity
    - ⚠️ Low average spread and spread volatility

    The higher the score, the more promising the pool is for stable MM profits.
    """)

with col_right:
    st.markdown("#### 🏆 Top 20 Pools")
    st.dataframe(pool_stats_normalized.sort_values("mm_score", ascending=False)[
        ["blockchain", "dex", "pool", "mm_score"]
    ].head(20).reset_index(drop=True), use_container_width=True)


st.markdown("---")


# __________________ Part4: Boxplot ______________________________________________________________________

st.subheader("📊 Spread Distribution by DEX")

col_text3, col_chart3 = st.columns([1, 1])

with col_chart3:
    box_plot = alt.Chart(filtered_data).mark_boxplot(extent='min-max').encode(
        x=alt.X("dex:N", title="DEX"),
        y=alt.Y("Spread", title="Spread (%)", scale=alt.Scale(type='log')),
        color="dex:N"
    ).properties(
        height=400,
        width=600
    )
    st.altair_chart(box_plot, use_container_width=True)

with col_text3:
    st.markdown("### What to look for?")
    st.markdown("""
    - **Goal**: Compare how well different DEXs manage slippage (spread).
    - **Look for**: DEXs with higher median or wider spread distribution; these may indicate weaker liquidity providers, offering **market making opportunities**.
    """)

st.markdown("---")
#----
st.markdown("Contact me:")
st.markdown("Email: bellabahramii@gmail.com")
st.markdown("Youtube: https://www.youtube.com/@bella_trickss")
st.markdown("Github: https://github.com/bellatrix-ds")



