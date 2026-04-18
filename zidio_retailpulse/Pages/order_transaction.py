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
# ORDER ANALYSIS
# =========================================
orders = (
    df.groupby('Invoice')['Quantity']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# =========================================
# PLOT GRAPH
# =========================================
plt.figure(figsize=(12,6))

orders.plot(kind='bar', color='teal')

plt.title("Top 10 Orders by Quantity")
plt.xlabel("Invoice")
plt.ylabel("Total Quantity")

plt.xticks(rotation=45)

# =========================================
# SAVE GRAPH
# =========================================
plt.tight_layout()
plt.savefig("outputs/order_analysis.png")

plt.close()

print("Order & Transaction graph saved successfully")