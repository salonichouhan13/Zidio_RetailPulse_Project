import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="RetailPulse Dashboard", layout="wide")

# -----------------------------
# UI STYLE (MODERN)
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #F1F5F9;
}
.block-container {
    padding: 2rem;
}
h1 {
    color: #111827;
    font-weight: 700;
}
.stMetric {
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    color: white !important;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.stPlotlyChart {
    background: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

st.title("📊 RetailPulse Analytics Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "..", "DATASET", "feature_engineered_data.csv")

df = pd.read_csv(data_path)

# Clean columns
df.columns = df.columns.str.strip()

# Fix column names
rename_map = {
    'Customer ID': 'CustomerID',
    'Invoice No': 'InvoiceNo'
}
df.rename(columns=rename_map, inplace=True)

# Date convert
if 'InvoiceDate' in df.columns:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# -----------------------------
# SIDEBAR FILTER
# -----------------------------
st.sidebar.header(" Filters")

if 'Country' in df.columns:
    country = st.sidebar.selectbox(
        "Select Country",
        ["All"] + list(df['Country'].dropna().unique())
    )
    if country != "All":
        df = df[df['Country'] == country]

# -----------------------------
# KPI CARDS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

# Auto detect invoice column
invoice_col = None
for col in df.columns:
    if "invoice" in col.lower():
        invoice_col = col
        break

total_sales = df['TotalAmount'].sum() if 'TotalAmount' in df.columns else 0
orders = df[invoice_col].nunique() if invoice_col else 0
customers = df['CustomerID'].dropna().nunique() if 'CustomerID' in df.columns else 0
quantity = df['Quantity'].sum() if 'Quantity' in df.columns else 0

col1.metric(" Total Sales", f"{total_sales:,.0f}")
col2.metric("Orders", orders)
col3.metric(" Customers", customers)
col4.metric(" Quantity", int(quantity))

st.markdown("---")

# -----------------------------
# SALES + QUANTITY
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Sales Trend")
    if 'InvoiceDate' in df.columns and 'TotalAmount' in df.columns:
        monthly = df.set_index('InvoiceDate')['TotalAmount'].resample('ME').sum()
        fig = px.line(monthly, color_discrete_sequence=["#6366F1"])
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader(" Quantity Trend")
    if 'InvoiceDate' in df.columns and 'Quantity' in df.columns:
        qty = df.set_index('InvoiceDate')['Quantity'].resample('ME').sum()
        fig = px.line(qty, color_discrete_sequence=["#22C55E"])
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# CHURN + SEGMENTATION
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Customer Churn")
    if 'CustomerID' in df.columns:
        churn_df = df.dropna(subset=['CustomerID'])
        churn = churn_df.groupby('CustomerID')['TotalAmount'].sum().reset_index()
        churn['Churn'] = churn['TotalAmount'].apply(
            lambda x: "Churn" if x < 1000 else "Active"
        )
        fig = px.pie(
            churn,
            names='Churn',
            color='Churn',
            color_discrete_map={
                "Churn": "#EF4444",
                "Active": "#22C55E"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("👥 Customer Segments")
    if 'CustomerID' in df.columns:
        cust = df.dropna(subset=['CustomerID']).groupby('CustomerID').agg({
            'TotalAmount': 'sum',
            'Quantity': 'sum'
        }).reset_index()

        if len(cust) > 3:
            kmeans = KMeans(n_clusters=3, random_state=42)
            cust['Segment'] = kmeans.fit_predict(cust[['TotalAmount', 'Quantity']])

            fig = px.scatter(
                cust,
                x='TotalAmount',
                y='Quantity',
                color=cust['Segment'].astype(str),
                color_discrete_sequence=["#6366F1", "#22C55E", "#F59E0B"]
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# FORECAST + TOP PRODUCTS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Demand Forecast")
    if 'InvoiceDate' in df.columns:
        demand = df.set_index('InvoiceDate')['Quantity'].resample('ME').sum()

        if len(demand) > 10:
            model = ARIMA(demand, order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=6)

            fig = px.line(demand, color_discrete_sequence=["#6366F1"])
            fig.add_scatter(
                x=forecast.index,
                y=forecast,
                mode='lines',
                name='Forecast',
                line=dict(color="#F59E0B", width=3)
            )
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader(" Top Products")
    if 'Description' in df.columns:
        top_products = df.groupby('Description')['Quantity'].sum().reset_index()
        top_products = top_products.sort_values(by='Quantity', ascending=False).head(10)

        fig = px.bar(
            top_products,
            x='Quantity',
            y='Description',
            orientation='h',
            color='Quantity',
            color_continuous_scale=['#6366F1', '#22C55E']
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# EXTRA GRAPHS (FROM OUTPUTS)
# -----------------------------
st.markdown("---")
st.subheader(" Additional Insights")

col1, col2 = st.columns(2)

with col1:
    if 'Country' in df.columns:
        country_sales = df.groupby('Country')['TotalAmount'].sum().reset_index()
        fig = px.bar(country_sales, x='Country', y='TotalAmount', color='TotalAmount')
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if 'CustomerID' in df.columns:
        top_customers = df.groupby('CustomerID')['TotalAmount'].sum().reset_index().head(10)
        fig = px.bar(top_customers, x='CustomerID', y='TotalAmount', color='TotalAmount')
        st.plotly_chart(fig, use_container_width=True)