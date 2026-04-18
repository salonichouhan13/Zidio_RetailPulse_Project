import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------
# 1. Paths
# ---------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "..", "DATASET", "feature_engineered_data.csv")
output_path = os.path.join(base_path, "outputs")

os.makedirs(output_path, exist_ok=True)

# ---------------------------
# 2. Load Data
# ---------------------------
df = pd.read_csv(data_path)

# ---------------------------
# 3. Demand per Product
# ---------------------------
product_demand = df.groupby('Description')['Quantity'].sum().reset_index()

# ---------------------------
# 4. Assumptions
# ---------------------------
lead_time = 5
ordering_cost = 50
holding_cost = 10

# ---------------------------
# 5. Calculations
# ---------------------------
product_demand['Daily_Demand'] = product_demand['Quantity'] / 30

# Reorder Point
product_demand['Reorder_Point'] = product_demand['Daily_Demand'] * lead_time

# Safety Stock
product_demand['Safety_Stock'] = product_demand['Daily_Demand'] * 2

# EOQ
D = product_demand['Quantity']
product_demand['EOQ'] = np.sqrt((2 * D * ordering_cost) / holding_cost)

# Cost Calculations
product_demand['Ordering_Cost_Total'] = (D / product_demand['EOQ']) * ordering_cost
product_demand['Holding_Cost_Total'] = (product_demand['EOQ'] / 2) * holding_cost

product_demand['Total_Inventory_Cost'] = (
    product_demand['Ordering_Cost_Total'] + 
    product_demand['Holding_Cost_Total']
)

# ---------------------------
# 6. Save CSV
# ---------------------------
csv_path = os.path.join(output_path, "advanced_inventory_results.csv")
product_demand.to_csv(csv_path, index=False)

# ---------------------------
# 7. Summary File
# ---------------------------
top_products = product_demand.sort_values(by='Quantity', ascending=False).head(5)

with open(os.path.join(output_path, "inventory_summary.txt"), "w") as f:
    f.write("ADVANCED INVENTORY OPTIMIZATION\n\n")
    f.write("Top Products:\n")
    f.write(str(top_products))

# ---------------------------
# 8. GRAPHS
# ---------------------------

# Top Demand Products
top10 = product_demand.sort_values(by='Quantity', ascending=False).head(10)

plt.figure()
plt.barh(top10['Description'], top10['Quantity'])
plt.title("Top 10 Products by Demand")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "top_products.png"))
plt.close()

# EOQ vs Demand
plt.figure()
plt.scatter(product_demand['Quantity'], product_demand['EOQ'])
plt.xlabel("Demand")
plt.ylabel("EOQ")
plt.title("EOQ vs Demand")
plt.savefig(os.path.join(output_path, "eoq_vs_demand.png"))
plt.close()

# Cost Analysis
top_cost = product_demand.sort_values(by='Total_Inventory_Cost', ascending=False).head(10)

plt.figure()
plt.bar(top_cost['Description'], top_cost['Total_Inventory_Cost'])
plt.xticks(rotation=90)
plt.title("Top 10 Products by Inventory Cost")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "cost_analysis.png"))
plt.close()

# ---------------------------
# 9. Output
# ---------------------------
print("Advanced Inventory Optimization Done!")
print(" CSV:", csv_path)