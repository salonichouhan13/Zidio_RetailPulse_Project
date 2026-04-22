import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="RetailPulse Dashboard", layout="wide")

# -----------------------------
# CUSTOM CSS (LIGHT THEME)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #F5F7FA;
}
h1, h2, h3 {
    color: #2E3A59;
}
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.title(" RetailPulse Analytics Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
import os

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "..", "DATASET", "feature_engineered_data.csv")

df = pd.read_csv(data_path)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header(" Filters")

country = st.sidebar.selectbox("Select Country", ["All"] + list(df['Country'].unique()))

if country != "All":
    df = df[df['Country'] == country]

# -----------------------------
# KPI CARDS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", f"{df['TotalAmount'].sum():,.0f}")
col2.metric("Orders", df['InvoiceNo'].nunique())
col3.metric("Customers", df['CustomerID'].nunique())
col4.metric("Quantity", int(df['Quantity'].sum()))

st.markdown("---")

# -----------------------------
# ROW 1 (Sales + Quantity Trend)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales Trend")
    monthly = df.set_index('InvoiceDate')['TotalAmount'].resample('ME').sum()

    fig = px.line(
        monthly,
        color_discrete_sequence=["#4CAF50"]
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Quantity Trend")
    qty = df.set_index('InvoiceDate')['Quantity'].resample('ME').sum()

    fig = px.line(
        qty,
        color_discrete_sequence=["#2196F3"]
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# ROW 2 (Churn + Segmentation)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Churn")

    churn = df.groupby('CustomerID')['TotalAmount'].sum().reset_index()
    churn['Churn'] = churn['TotalAmount'].apply(lambda x: "Churn" if x < 1000 else "Active")

    fig = px.pie(
        churn,
        names='Churn',
        color_discrete_sequence=["#FF6B6B", "#4CAF50"]
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Customer Segments")

    cust = df.groupby('CustomerID').agg({
        'TotalAmount': 'sum',
        'Quantity': 'sum'
    }).reset_index()

    kmeans = KMeans(n_clusters=3, random_state=42)
    cust['Segment'] = kmeans.fit_predict(cust[['TotalAmount', 'Quantity']])

    fig = px.scatter(
        cust,
        x='TotalAmount',
        y='Quantity',
        color=cust['Segment'].astype(str),
        color_discrete_sequence=["#4CAF50", "#2196F3", "#FFC107"]
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# ROW 3 (Forecast + Inventory)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demand Forecast")

    demand = df.set_index('InvoiceDate')['Quantity'].resample('ME').sum()

    model = ARIMA(demand, order=(1,1,1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=6)

    fig = px.line(
        demand,
        color_discrete_sequence=["#2196F3"]
    )

    fig.add_scatter(
        x=forecast.index,
        y=forecast,
        mode='lines',
        name='Forecast',
        line=dict(color="#FF9800")
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top Products")

    inv = df.groupby('Description')['Quantity'].sum().reset_index()
    top_products = inv.sort_values(by='Quantity', ascending=False).head(10)

    fig = px.bar(
        top_products,
        x='Quantity',
        y='Description',
        orientation='h',
        color='Quantity',
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)