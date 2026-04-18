# =========================================
#  IMPORT LIBRARIES
# =========================================
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================
# CREATE OUTPUT FOLDER
# =========================================
os.makedirs("outputs", exist_ok=True)

# =========================================
#LOAD DATA
# =========================================
df = pd.read_csv("cleaned_data.csv")

# =========================================
# TOP CUSTOMERS
# =========================================
top_customers = (
    df.groupby("Customer ID")['TotalPrice']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# =========================================
# PLOT GRAPH (COLOR CHANGE)
# =========================================
plt.figure(figsize=(12,6))

#
top_customers.plot(kind='bar', color='orange')   

plt.title("Top 10 Customers by Spending")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")

plt.xticks(rotation=45)

# =========================================
#  SAVE GRAPH
# =========================================
plt.tight_layout()
plt.savefig("outputs/customer_analysis.png")

plt.close()

print(" Customer Analysis graph saved with new color ")