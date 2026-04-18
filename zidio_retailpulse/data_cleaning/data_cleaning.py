# =========================================
#  IMPORT LIBRARIES
# =========================================
import numpy as np
import pandas as pd

# =========================================
#  LOAD DATASET
# =========================================
df = pd.read_csv("zidio_retailpulse/DATASET/online_retail_ii.csv", encoding='latin1')

print("✅ Dataset Loaded Successfully")
print("Shape Before Cleaning:", df.shape)

# =========================================
#  BASIC INFO SAVE
# =========================================
df.head().to_csv("raw_head.csv", index=False)
df.describe().to_csv("raw_summary.csv")

missing = df.isnull().sum()
missing.to_csv("missing_values.csv")

# =========================================
#  DATA CLEANING
# =========================================

# 1. Remove missing values
df.dropna(inplace=True)

#  REMOVE THIS LINE (गलत है)
# df.fillna(inplace=True)

# 2. Remove invalid values
df = df[df['Quantity'] > 0]
df = df[df['Price'] > 0]

# 3. Remove duplicates
df.drop_duplicates(inplace=True)

# 4. Convert date column
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 5. Create Total Price
df['TotalPrice'] = df['Quantity'] * df['Price']

# =========================================
#  FEATURE ENGINEERING
# =========================================
df['Month'] = df['InvoiceDate'].dt.month
df['Year'] = df['InvoiceDate'].dt.year

# =========================================
#  OUTLIER REMOVAL
# =========================================
q1 = df['TotalPrice'].quantile(0.01)
q2 = df['TotalPrice'].quantile(0.99)

df = df[(df['TotalPrice'] >= q1) & (df['TotalPrice'] <= q2)]

print("Shape After Cleaning:", df.shape)

# =========================================
# SAVE CLEANED DATA
# =========================================
df.to_csv("cleaned_data.csv", index=False)

print(" cleaned_data.csv saved")

# =========================================
#  SAVE SUMMARY
# =========================================
df.describe().to_csv("cleaned_summary.csv")

# =========================================
#  SAVE UNIQUE INFO
# =========================================
unique_info = pd.DataFrame({
    "Metric": ["Customers", "Products", "Countries", "Invoices"],
    "Count": [
        df["Customer ID"].nunique(),
        df["StockCode"].nunique(),
        df["Country"].nunique(),
        df["Invoice"].nunique()
    ]
})

unique_info.to_csv("unique_counts.csv", index=False)

print("All results saved successfully ")