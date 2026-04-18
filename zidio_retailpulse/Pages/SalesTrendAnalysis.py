# =========================================
#  IMPORT LIBRARIES
# =========================================
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================
#  CREATE OUTPUT FOLDER
# =========================================
os.makedirs("outputs", exist_ok=True)

# =========================================
#  LOAD CLEANED DATA
# =========================================
df = pd.read_csv("cleaned_data.csv")

# =========================================
#  DATE PROCESSING
# =========================================
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Date'] = df['InvoiceDate'].dt.date

# =========================================
#  GROUP DATA
# =========================================
trend = df.groupby('Date')['TotalPrice'].sum()

# =========================================
#  PLOT GRAPH
# =========================================
plt.figure(figsize=(10,5))
trend.plot()

plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")

# =========================================
#  SAVE GRAPH
# =========================================
plt.savefig("outputs/sales_trend.png")

# =========================================
#  SHOW GRAPH
# =========================================
plt.show()

print(" Graph saved in outputs folder")
plt.savefig("outputs/sales_trend.png")
