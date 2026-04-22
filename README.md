#  RetailPulse - End-to-End Retail Analytics Project

## 📌 Overview

RetailPulse is a complete data analytics and machine learning project that analyzes retail data to generate insights, forecast demand, predict customer churn, and optimize inventory.
It covers the full pipeline from raw data processing to an interactive dashboard.

---

##  Key Features

*  Sales Analysis & Data Cleaning
*  Demand Forecasting (Time Series - Prophet / ARIMA)
*  Customer Segmentation (RFM Analysis)
* Churn Prediction (Machine Learning Model)
*  Inventory Optimization (EOQ Model)
*  Interactive Dashboard (Streamlit)

---

##  Project Structure

ZIDIO_1ST/
│
├── zidio_retailpulse/
│   ├── data_cleaning/
│   ├── feature_engineering/
│   ├── customer_segmentation/
│   ├── Churn_Prediction/
│   ├── Demand_forcasting/
│   ├── inventory_optimization/
│   ├── Dashboard_Streamlit/
│   │   └── app.py
│   ├── DATASET/
│   │   └── feature_engineered_data.csv
│   └── outputs/
│
├── cleaned_summary.csv
├── missing_values.csv
├── raw_head.csv
├── raw_summary.csv
├── unique_counts.csv

---

##  Technologies Used

* Python 
* Pandas
* NumPy
* Matplotlib
* Plotly
* Scikit-learn
* Prophet / Statsmodels
* Streamlit

---

## Dashboard Features

* Interactive Filters (Date & Country)
* KPI Cards (Sales, Orders, Customers)
* Monthly Sales Trends
* Demand Forecast Visualization
* Customer Churn Analysis
* Inventory Optimization Insights

---

# How to Run

### 1️ Install dependencies

pip install -r requirements.txt

### 2️ Run Dashboard

cd zidio_retailpulse/Dashboard_Streamlit
streamlit run app.py

---

##  Live Demo

👉 (Add your deployed link here after deployment)

##  Results

* Identified top-selling products
* Analyzed sales trends over time
* Predicted future demand
* Detected churn-prone customers
* Optimized inventory using EOQ

---

##  Conclusion

This project demonstrates a complete data analytics pipeline:

Data Cleaning → Feature Engineering → Machine Learning → Forecasting → Dashboard Visualization

---

## Author

Saloni Chouhan
(Data Analytics / Data Science Project)
