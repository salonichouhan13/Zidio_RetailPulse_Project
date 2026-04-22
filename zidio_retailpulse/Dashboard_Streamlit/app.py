#--how to run dashboard---
#--give the command in the terminal as given below
# 1---  cd zidio_retailpulse/Dashboard_Streamlit
# 2----  streamlit run app.py


import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="RetailPulse", layout="wide")

# ---------------------------
#  CUSTOM COLORS
# ---------------------------
st.markdown("""
<style>
.main {background-color: #0E1117; color: white;}
h1, h2, h3 {color: #00C9A7;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
#  LOAD DATA
# ---------------------------
data_path = r"C:\Users\HP\OneDrive\Desktop\ZIDIO_1ST\zidio_retailpulse\DATASET\feature_engineered_data.csv"
df = pd.read_csv(data_path)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ---------------------------
#  SIDEBAR FILTERS
# ---------------------------
st.sidebar.title("Filters")

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df['InvoiceDate'].min(), df['InvoiceDate'].max()]
)

country = st.sidebar.selectbox(
    "Select Country",
    ["All"] + list(df['Country'].dropna().unique())
)

# Apply filters
if len(date_range) == 2:
    df = df[(df['InvoiceDate'] >= pd.to_datetime(date_range[0])) &
            (df['InvoiceDate'] <= pd.to_datetime(date_range[1]))]

if country != "All":
    df = df[df['Country'] == country]

# ---------------------------
#  TITLE
# ---------------------------
st.markdown("<h1 style='text-align:center;'> RetailPulse Dashboard</h1>", unsafe_allow_html=True)

# ---------------------------
#  KPI CARDS
# ---------------------------
col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"""
<div style="background:#FF6B6B;padding:20px;border-radius:10px;text-align:center">
<h4>Total Sales</h4>
<h2>{round(df['TotalAmount'].sum(),2)}</h2>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div style="background:#4ECDC4;padding:20px;border-radius:10px;text-align:center">
<h4>Orders</h4>
<h2>{df['Invoice'].nunique()}</h2>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div style="background:#FFD93D;padding:20px;border-radius:10px;text-align:center">
<h4>Customers</h4>
<h2>{df['Customer ID'].nunique()}</h2>
</div>
""", unsafe_allow_html=True)

col4.markdown(f"""
<div style="background:#6C5CE7;padding:20px;border-radius:10px;text-align:center">
<h4>Quantity</h4>
<h2>{int(df['Quantity'].sum())}</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
#  TABS
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    " Sales",
    " Forecast",
    " Customers",
    " Inventory"
])

# ---------------------------
# SALES
# ---------------------------
with tab1:
    monthly = df.set_index('InvoiceDate')['TotalAmount'].resample('M').sum()

    fig = px.line(monthly, title="Monthly Sales Trend",
                  color_discrete_sequence=['#00C9A7'])
    st.plotly_chart(fig, use_container_width=True)

    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

    fig2 = px.bar(top_products, orientation='h',
                  color=top_products.values,
                  color_continuous_scale='viridis',
                  title="Top Products")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
#  FORECAST
# ---------------------------
with tab2:
    monthly = df.set_index('InvoiceDate')['TotalAmount'].resample('M').sum()
    forecast = monthly.rolling(3).mean()

    forecast_df = pd.DataFrame({
        "Actual": monthly,
        "Forecast": forecast
    })

    fig = px.line(forecast_df, title="Forecast vs Actual")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
#  CHURN
# ---------------------------
with tab3:
    churn = df.groupby('Customer ID')['InvoiceDate'].max()
    recency = (df['InvoiceDate'].max() - churn).dt.days

    churn_flag = recency > 90
    churn_df = churn_flag.value_counts().rename(index={True: "Churned", False: "Active"})

    fig = px.pie(values=churn_df.values, names=churn_df.index,
                 title="Customer Churn")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
#  INVENTORY
# ---------------------------
with tab4:
    product = df.groupby('Description')['Quantity'].sum().reset_index()
    product['EOQ'] = (2 * product['Quantity'] * 50 / 10) ** 0.5

    top = product.sort_values(by='EOQ', ascending=False).head(10)

    fig = px.bar(top, x='EOQ', y='Description',
                 orientation='h',
                 color='EOQ',
                 color_continuous_scale='plasma',
                 title="Top EOQ Products")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
#  FOOTER
# ---------------------------
st.markdown("---")
