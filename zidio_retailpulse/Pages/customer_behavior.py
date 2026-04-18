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
# CUSTOMER ORDER FREQUENCY
# =========================================
customer_orders = (
    df.groupby("Customer ID")['Invoice']
    .count()
    .sort_values(ascending=False)
    .head(10)
)

# =========================================
# PLOT GRAPH
# =========================================
plt.figure(figsize=(12,6))

customer_orders.plot(kind='bar', color='red')

plt.title("Top 10 Customers by Number of Orders")
plt.xlabel("Customer ID")
plt.ylabel("Number of Orders")

plt.xticks(rotation=45)

# =========================================
# SAVE GRAPH
# =========================================
plt.tight_layout()
plt.savefig("outputs/customer_behavior.png")

plt.close()

print("Customer Behavior graph saved successfully")