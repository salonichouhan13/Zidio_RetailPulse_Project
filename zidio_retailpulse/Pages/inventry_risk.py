# =========================================
# IMPORT LIBRARIES
# =========================================
import pandas as pd
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
# INVENTORY RISK LOGIC
# =========================================
# Low stock products (Quantity < 5)
low_stock = df[df['Quantity'] < 5]

# =========================================
# SAVE RESULT
# =========================================
low_stock.to_csv("outputs/inventory_risk.csv", index=False)

# =========================================
# PRINT SAMPLE
# =========================================
print("Top 10 Low Stock Products:\n")
print(low_stock[['Description', 'Quantity']].head(10))

print("\nInventory Risk file saved successfully")