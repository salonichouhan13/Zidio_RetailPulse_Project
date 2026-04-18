import pandas as pd
import os

# ---------------------------
# 1. Fix Paths
# ---------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "..", "DATASET", "cleanData.csv")
save_path = os.path.join(base_path, "..", "DATASET")
insight_path = os.path.join(base_path, "..", "outputs", "feature_eng_insights")

# create folders if not exist
os.makedirs(save_path, exist_ok=True)
os.makedirs(insight_path, exist_ok=True)

# ---------------------------
# 2. Load Data
# ---------------------------
df = pd.read_csv(data_path)

# ---------------------------
# 3. Feature Engineering
# ---------------------------
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['Month'] = df['InvoiceDate'].dt.month
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
df['Hour'] = df['InvoiceDate'].dt.hour
df['Is_Weekend'] = df['InvoiceDate'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

# Total Sales
df['TotalAmount'] = df['Quantity'] * df['Price']

# Customer Features
df['Customer_Total_Spend'] = df.groupby('Customer ID')['TotalAmount'].transform('sum')
df['Customer_Frequency'] = df.groupby('Customer ID')['Invoice'].transform('nunique')

# Product Demand
df['Product_Demand'] = df.groupby('StockCode')['Quantity'].transform('sum')

# ---------------------------
# 4. Save Feature Data
# ---------------------------
df.to_csv(os.path.join(save_path, "feature_engineered_data.csv"), index=False)

# ---------------------------
# 5. Generate Insights
# ---------------------------
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(5)
monthly_sales = df.groupby('Month')['TotalAmount'].sum()
top_customers = df.groupby('Customer ID')['TotalAmount'].sum().sort_values(ascending=False).head(5)
weekend_sales = df.groupby('Is_Weekend')['TotalAmount'].sum()

# ---------------------------
# 6. Save Insights
# ---------------------------
with open(os.path.join(insight_path, "insights.txt"), "w") as f:
    f.write(" RETAILPULSE PROJECT INSIGHTS\n\n")
    
    f.write("Top Selling Products:\n")
    f.write(str(top_products) + "\n\n")
    
    f.write("Monthly Sales:\n")
    f.write(str(monthly_sales) + "\n\n")
    
    f.write("Top Customers:\n")
    f.write(str(top_customers) + "\n\n")
    
    f.write("Weekend vs Weekday Sales:\n")
    f.write(str(weekend_sales) + "\n\n")

# Ensure folder exists
import os
os.makedirs(save_path, exist_ok=True)

# Full file path
file_path = os.path.join(save_path, "feature_engineered_data.csv")

# Save
df.to_csv(file_path, index=False)

# Debug print
print("Saved successfully at:", file_path)