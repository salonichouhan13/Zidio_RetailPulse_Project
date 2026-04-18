# =========================================
#  IMPORT LIBRARIES
# =========================================
import pandas as pd
import os

# =========================================
# CREATE OUTPUT FOLDER
# =========================================
os.makedirs("outputs", exist_ok=True)

# =========================================
#  LOAD DATA
# =========================================
df = pd.read_csv("cleaned_data.csv")

# =========================================
# CALCULATIONS
# =========================================
total_sales = df['TotalPrice'].sum()
total_orders = df['Invoice'].nunique()
total_customers = df["Customer ID"].nunique()

# =========================================
# CREATE SUMMARY TABLE
# =========================================
summary = pd.DataFrame({
    "Metric": ["Total Sales", "Total Orders", "Total Customers"],
    "Value": [total_sales, total_orders, total_customers]
})

# =========================================
#  SAVE RESULT
# =========================================
summary.to_csv("outputs/executive_summary.csv", index=False)

# =========================================
#  PRINT RESULT
# =========================================
print("\n EXECUTIVE SUMMARY:\n")
print(summary)

print("\n Summary saved in outputs folder ")