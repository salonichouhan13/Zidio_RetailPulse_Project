# =========================================
# IMPORT LIBRARIES
# =========================================
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================
# CREATE OUTPUT FOLDER
# =========================================
os.makedirs("outputs", exist_ok=True)

# =========================================
# LOAD DATA
# =========================================
df = pd.read_csv("cleaned_data.csv")

# =========================================
# DATE PROCESSING
# =========================================
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.month

# =========================================
# MONTHLY SALES
# =========================================
monthly_sales = df.groupby('Month')['TotalPrice'].sum()

# =========================================
# PLOT GRAPH
# =========================================
plt.figure(figsize=(10,5))

monthly_sales.plot(kind='line', marker='o', color='purple')

plt.title("Monthly Sales Analysis")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.xticks(range(1,13))

# =========================================
# SAVE GRAPH
# =========================================
plt.tight_layout()
plt.savefig("outputs/monthly_sales.png")

plt.close()

print("Monthly Sales graph saved successfully")