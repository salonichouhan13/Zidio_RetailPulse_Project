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
# DARK UI STYLE
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
.block-container {
    padding: 1.5rem 2rem;
}
h1, h2, h3 {
    color: #f1f5f9;
}

/* KPI BOX */
.stMetric {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    color: white;
}

/* CHART BOX */
.stPlotlyChart {
    background: #1e293b;
    padding: 15px;
    border-radius: 12px;
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
df.columns = df.columns.str.strip()

# Rename columns if needed
rename_map = {
    'Customer ID': 'CustomerID',
    'Invoice No': 'InvoiceNo'
}
df.rename(columns=rename_map, inplace=True)

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# -----------------------------
# SIDEBAR FILTER
# -----------------------------
st.sidebar.header("🔎 Filters")

if 'Country' in df.columns:
    country = st.sidebar.selectbox(
        "Select Country",
        ["All"] + sorted(df['Country'].dropna().unique())
    )
    if country != "All":
        df = df[df['Country'] == country]

# -----------------------------
# KPI CARDS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

total_sales = df['TotalAmount'].sum()

orders = 0
for col in df.columns:
    if "invoice" in col.lower():
        orders = df[col].dropna().nunique()
        break

customers = df['CustomerID'].dropna().nunique()
quantity = df['Quantity'].sum()

col1.metric("💰 Total Sales", f"{total_sales:,.0f}")
col2.metric("🧾 Orders", orders)
col3.metric("👥 Customers", customers)
col4.metric("📦 Quantity", int(quantity))

st.markdown("---")

# -----------------------------
# SALES & QUANTITY
# -----------------------------
col1, col2 = st.columns(2)

monthly = df.set_index('InvoiceDate')['TotalAmount'].resample('ME').sum()
qty = df.set_index('InvoiceDate')['Quantity'].resample('ME').sum()

with col1:
    st.subheader("📈 Sales Trend")
    fig = px.line(monthly, color_discrete_sequence=["#6366F1"])
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📦 Quantity Trend")
    fig = px.line(qty, color_discrete_sequence=["#22C55E"])
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# CHURN & SEGMENTATION
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚠️ Customer Churn")
    churn_df = df.dropna(subset=['CustomerID'])

    churn = churn_df.groupby('CustomerID')['TotalAmount'].sum().reset_index()
    churn['Churn'] = churn['TotalAmount'].apply(lambda x: "Churn" if x < 1000 else "Active")

    fig = px.pie(
        churn,
        names='Churn',
        color='Churn',
        color_discrete_map={"Churn": "#ef4444", "Active": "#22c55e"}
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("👥 Customer Segments")

    cust = churn_df.groupby('CustomerID').agg({
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
        color_discrete_sequence=["#6366F1", "#22C55E", "#F59E0B"]
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# FORECAST & TOP PRODUCTS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 Demand Forecast")

    demand = df.set_index('InvoiceDate')['Quantity'].resample('ME').sum()

    if len(demand) > 10:
        model = ARIMA(demand, order=(1,1,1))
        forecast = model.fit().forecast(6)

        fig = px.line(demand, color_discrete_sequence=["#6366F1"])
        fig.add_scatter(x=forecast.index, y=forecast, mode='lines', name="Forecast",
                        line=dict(color="#f59e0b"))

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🏆 Top Products")

    top_products = df.groupby('Description')['Quantity'].sum().reset_index()
    top_products = top_products.sort_values(by='Quantity', ascending=False).head(10)

    fig = px.bar(
        top_products,
        x='Quantity',
        y='Description',
        orientation='h',
        color='Quantity',
        color_continuous_scale='Blues'
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# EXTRA GRAPHS (ADVANCED LOOK)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌍 Country Sales")

    country_sales = df.groupby('Country')['TotalAmount'].sum().reset_index()
    fig = px.bar(country_sales.sort_values(by='TotalAmount', ascending=False).head(10),
                 x='Country', y='TotalAmount', color='TotalAmount')
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🕒 Hourly Sales")

    df['Hour'] = df['InvoiceDate'].dt.hour
    hourly = df.groupby('Hour')['TotalAmount'].sum().reset_index()

    fig = px.line(hourly, x='Hour', y='TotalAmount')
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)