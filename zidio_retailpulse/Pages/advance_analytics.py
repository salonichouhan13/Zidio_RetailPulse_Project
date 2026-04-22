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
df = pd.read_csv("zidio_retailpulse/DATASET/cleanData.csv")

# =========================================
# DATE CONVERSION
# =========================================
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# =========================================
# 1️ SALES DISTRIBUTION
# =========================================
plt.figure(figsize=(10,5))

plt.hist(df['TotalPrice'], bins=50, color='teal')

plt.title("Sales Distribution")
plt.xlabel("Total Price")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("outputs/sales_distribution.png")
plt.close()

# =========================================
# 2️ BASKET SIZE
# =========================================
basket = df.groupby('Invoice')['Quantity'].sum()

plt.figure(figsize=(10,5))

plt.hist(basket, bins=40, color='orange')

plt.title("Basket Size Distribution")
plt.xlabel("Quantity per Order")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("outputs/basket_size.png")
plt.close()

# =========================================
# 3️ HOURLY SALES
# =========================================
df['Hour'] = df['InvoiceDate'].dt.hour
hourly = df.groupby('Hour')['TotalPrice'].sum()

plt.figure(figsize=(10,5))

plt.plot(hourly, marker='o', color='purple')

plt.title("Hourly Sales Pattern")
plt.xlabel("Hour")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig("outputs/hourly_sales.png")
plt.close()

# =========================================
# 4️ COUNTRY SALES
# =========================================
country_sales = (
    df.groupby('Country')['TotalPrice']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))

country_sales.plot(kind='bar', color='green')

plt.title("Top Countries by Sales")
plt.xlabel("Country")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("outputs/country_sales.png")
plt.close()

# =========================================
# 5️ TOP CUSTOMERS
# =========================================
top_customers = (
    df.groupby('Customer ID')['TotalPrice']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))

top_customers.plot(kind='bar', color='red')

plt.title("Top Customers")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("outputs/top_customers.png")
plt.close()

# =========================================
# DONE
# =========================================
