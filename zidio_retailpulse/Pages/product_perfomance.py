# =========================================
# IMPORT LIBRARIES
# =========================================
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================
#  CREATE OUTPUT FOLDER
# =========================================
os.makedirs("outputs", exist_ok=True)

# =========================================
#  LOAD DATA
# =========================================
df = pd.read_csv("cleaned_data.csv")

# =========================================
#  TOP PRODUCTS ANALYSIS
# =========================================
top_products = (
    df.groupby('Description')['Quantity']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# =========================================
#  PLOT GRAPH
# =========================================
plt.figure(figsize=(12,6))
top_products.plot(kind='bar')

plt.title("Top 10 Products by Quantity Sold")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")

# rotate labels (important for readability)
plt.xticks(rotation=45, ha='right')

# =========================================
#  SAVE GRAPH
# =========================================
plt.tight_layout()
plt.savefig("outputs/top_products.png")

#  no plt.show() (avoid error)
plt.close()

print(" Product Performance graph saved in outputs folder 🎯")