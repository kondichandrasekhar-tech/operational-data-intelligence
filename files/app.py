"""Optional Streamlit dashboard:  streamlit run app.py"""
import streamlit as st
from pathlib import Path
import odi

st.set_page_config(page_title="Operational Data Intelligence", layout="wide")
st.title("Operational Data Intelligence")

df = odi.add_kpis(odi.clean_data(odi.load_data(Path("data/operations.csv"))))
depts = st.sidebar.multiselect("Department", sorted(df["department"].unique()), default=list(df["department"].unique()))
df = df[df["department"].isin(depts)]

summary = odi.department_summary(df)
anomalies = odi.detect_anomalies(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders", f"{df['orders'].sum():,.0f}")
c2.metric("Completion rate", f"{df['completed'].sum() / df['orders'].sum():.1%}")
c3.metric("Profit", f"{df['profit'].sum():,.0f}")
c4.metric("Anomalies", len(anomalies))

st.subheader("Insights")
for i in odi.generate_insights(df, summary, anomalies):
    st.write("•", i)

st.subheader("Daily trend")
st.line_chart(df.groupby("date")[["orders", "profit"]].sum())
st.subheader("Department summary")
st.dataframe(summary)
st.subheader("Anomalies")
st.dataframe(anomalies)
