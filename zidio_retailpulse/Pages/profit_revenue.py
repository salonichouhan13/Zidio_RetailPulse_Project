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
df['Date'] = df['InvoiceDate'].dt.date

# =========================================
# REVENUE ANALYSIS
# =========================================
revenue = df.groupby('Date')['TotalPrice'].sum()

# =========================================
# PLOT GRAPH
# =========================================
plt.figure(figsize=(10,5))

revenue.plot(kind='line', color='brown')

plt.title("Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")

# =========================================
# SAVE GRAPH
# =========================================
plt.tight_layout()
plt.savefig("outputs/revenue_analysis.png")

plt.close()

print("Revenue Analysis graph saved successfully")