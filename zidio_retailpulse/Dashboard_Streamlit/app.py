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
# UI STYLE
# -----------------------------
st.markdown("""
<style>
.main { background-color: #F8FAFC; }
.block-container { padding: 1.5rem 2rem; }
h1, h2, h3 { color: #111827; }
.stMetric {
    background: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.06);
}
.stPlotlyChart {
    background: white;
    padding: 10px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
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

# Clean column names
df.columns = df.columns.str.strip()

# Auto rename
df.rename(columns={
    'Customer ID': 'CustomerID',
    'customer_id': 'CustomerID',
    'Invoice No': 'InvoiceNo',
    'invoice_no': 'InvoiceNo'
}, inplace=True)

# Convert date
if 'InvoiceDate' in df.columns:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# -----------------------------
# SIDEBAR
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

total_sales = df.get('TotalAmount', pd.Series()).sum()

# Auto detect orders column
orders = 0
for col in df.columns:
    if "invoice" in col.lower():
        orders = df[col].dropna().nunique()
        break

customers = df.get('CustomerID', pd.Series()).dropna().nunique()
quantity = df.get('Quantity', pd.Series()).sum()

col1.metric("💰 Total Sales", f"{total_sales:,.0f}")
col2.metric("🧾 Orders", orders)
col3.metric("👥 Customers", customers)
col4.metric("📦 Quantity", int(quantity))

st.markdown("---")

# -----------------------------
# SALES + QUANTITY
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Sales Trend")
    if 'InvoiceDate' in df.columns and 'TotalAmount' in df.columns:
        monthly = df.set_index('InvoiceDate')['TotalAmount'].resample('ME').sum()
        fig = px.line(monthly, color_discrete_sequence=["#6366F1"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📦 Quantity Trend")
    if 'InvoiceDate' in df.columns and 'Quantity' in df.columns:
        qty = df.set_index('InvoiceDate')['Quantity'].resample('ME').sum()
        fig = px.line(qty, color_discrete_sequence=["#10B981"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# CHURN + SEGMENTATION
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚠️ Customer Churn")
    if 'CustomerID' in df.columns and 'TotalAmount' in df.columns:
        churn_df = df.dropna(subset=['CustomerID'])
        churn = churn_df.groupby('CustomerID')['TotalAmount'].sum().reset_index()
        churn['Churn'] = churn['TotalAmount'].apply(
            lambda x: "Churn" if x < 1000 else "Active"
        )
        fig = px.pie(
            churn,
            names='Churn',
            color='Churn',
            color_discrete_map={"Churn": "#EF4444", "Active": "#10B981"}
        )
        fig.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("👥 Customer Segments")
    if 'CustomerID' in df.columns:
        cust_df = df.dropna(subset=['CustomerID'])
        cust = cust_df.groupby('CustomerID').agg({
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
                color_discrete_sequence=["#6366F1", "#10B981", "#F59E0B"]
            )
            fig.update_layout(paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# FORECAST + TOP PRODUCTS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 Demand Forecast")
    if 'InvoiceDate' in df.columns and 'Quantity' in df.columns:
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
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🏆 Top Products")
    if 'Description' in df.columns and 'Quantity' in df.columns:
        inv = df.groupby('Description')['Quantity'].sum().reset_index()
        top_products = inv.sort_values(by='Quantity', ascending=False).head(10)

        fig = px.bar(
            top_products,
            x='Quantity',
            y='Description',
            orientation='h',
            color='Quantity',
            color_continuous_scale=['#6366F1', '#10B981']
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)