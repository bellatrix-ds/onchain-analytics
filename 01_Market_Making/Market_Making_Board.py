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
    'https://raw.githubusercontent.com/bellatrix-ds/onchain-analytics/refs/heads/main/01_Market_Making/Market%20Making.csv',
    on_bad_lines='skip'
)

# --------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Curve Pool Dashboard")

# فیلترهای اولیه
col1, col2, col3 = st.columns(3)
with col1:
    selected_chain = st.selectbox("🔗 Select Chain", sorted(df['Chain'].unique()))
with col2:
    selected_dex = st.selectbox("📈 Select DEX", sorted(df['Dex'].unique()))
with col3:
    selected_type = st.selectbox("💱 Select Pool Type", sorted(df['Pool_Type'].unique()))

# فیلتر داده‌ها بر اساس انتخاب‌ها
filtered_df = df[
    (df['Chain'] == selected_chain) &
    (df['Dex'] == selected_dex)
]

# 📊 نمودار دایره‌ای: سهم بازار هر استخر بر اساس Share
fig1 = px.pie(
    filtered_df,
    values='Share',
    names='Pool_Name',
    title=f"📌 Market Share of Pools on {selected_dex} ({selected_chain})",
    hole=0.45
)

# 📈 درصد انواع استخرها
type_counts = filtered_df['Pool_Type'].value_counts(normalize=True) * 100
type_df = pd.DataFrame({
    'Type': type_counts.index,
    'Percentage': type_counts.values
})

fig2 = px.pie(
    type_df,
    names='Type',
    values='Percentage',
    title=f"🔍 Pool Type Distribution on {selected_dex}",
    hole=0.45
)

# نمایش نمودارها در کنار هم
col4, col5 = st.columns(2)
with col4:
    st.plotly_chart(fig1, use_container_width=True)
with col5:
    st.plotly_chart(fig2, use_container_width=True)

# 📋 نمایش دیتای فیلترشده
st.subheader("📄 Filtered Pool Table")
st.dataframe(filtered_df.reset_index(drop=True))

# ـــ

st.markdown("---")
st.caption("Contact me: bellabahramii@gmail.com")


