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
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
body { background-color: #F5F7FA; }
h1, h2, h3 { color: #2E3A59; }
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.title("📊 RetailPulse Analytics Dashboard")

# -----------------------------
# LOAD DATA (FIXED PATH)
# -----------------------------
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "..", "DATASET", "feature_engineered_data.csv")

df = pd.read_csv(data_path)

# CLEAN COLUMNS
df.columns = df.columns.str.strip()

# SAFE DATE
if 'InvoiceDate' in df.columns:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🔎 Filters")

if 'Country' in df.columns:
    country = st.sidebar.selectbox("Select Country", ["All"] + list(df['Country'].dropna().unique()))
    if country != "All":
        df = df[df['Country'] == country]

# -----------------------------
# KPI CARDS (SAFE)
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

# Total Sales
total_sales = df['TotalAmount'].sum() if 'TotalAmount' in df.columns else 0
col1.metric("Total Sales", f"{total_sales:,.0f}")

# Orders
if 'InvoiceNo' in df.columns:
    orders = df['InvoiceNo'].nunique()
elif 'Invoice' in df.columns:
    orders = df['Invoice'].nunique()
else:
    orders = len(df)
col2.metric("Orders", orders)

# Customers
customers = df['CustomerID'].nunique() if 'CustomerID' in df.columns else 0
col3.metric("Customers", customers)

# Quantity
quantity = int(df['Quantity'].sum()) if 'Quantity' in df.columns else 0
col4.metric("Quantity", quantity)

st.markdown("---")

# -----------------------------
# ROW 1
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales Trend")
    if 'InvoiceDate' in df.columns and 'TotalAmount' in df.columns:
        monthly = df.set_index('InvoiceDate')['TotalAmount'].resample('ME').sum()
        fig = px.line(monthly, color_discrete_sequence=["#4CAF50"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Quantity Trend")
    if 'InvoiceDate' in df.columns and 'Quantity' in df.columns:
        qty = df.set_index('InvoiceDate')['Quantity'].resample('ME').sum()
        fig = px.line(qty, color_discrete_sequence=["#2196F3"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# ROW 2
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Churn")
    if 'CustomerID' in df.columns and 'TotalAmount' in df.columns:
        churn = df.groupby('CustomerID')['TotalAmount'].sum().reset_index()
        churn['Churn'] = churn['TotalAmount'].apply(lambda x: "Churn" if x < 1000 else "Active")
        fig = px.pie(churn, names='Churn', color_discrete_sequence=["#FF6B6B", "#4CAF50"])
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Customer Segments")
    if 'CustomerID' in df.columns:
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
# ROW 3
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demand Forecast")
    if 'InvoiceDate' in df.columns and 'Quantity' in df.columns:
        demand = df.set_index('InvoiceDate')['Quantity'].resample('ME').sum()

        model = ARIMA(demand, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=6)

        fig = px.line(demand, color_discrete_sequence=["#2196F3"])
        fig.add_scatter(x=forecast.index, y=forecast, mode='lines', name='Forecast',
                        line=dict(color="#FF9800"))

        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top Products")
    if 'Description' in df.columns and 'Quantity' in df.columns:
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
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)