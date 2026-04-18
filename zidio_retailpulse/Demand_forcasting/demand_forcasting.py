import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# 1. Fix Paths
# ---------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "..", "DATASET", "feature_engineered_data.csv")
output_path = os.path.join(base_path, "outputs")

# create outputs folder
os.makedirs(output_path, exist_ok=True)

# ---------------------------
# 2. Load Data
# ---------------------------
df = pd.read_csv(data_path)

# convert date
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ---------------------------
# 3. Create Time Series (Monthly Sales)
# ---------------------------
df.set_index('InvoiceDate', inplace=True)

monthly_sales = df['TotalAmount'].resample('M').sum()

# ---------------------------
# 4. Forecast using Moving Average
# ---------------------------
forecast = monthly_sales.rolling(window=3).mean()

# ---------------------------
# 5. Save Forecast Data
# ---------------------------
forecast_df = pd.DataFrame({
    "Actual_Sales": monthly_sales,
    "Forecast_Sales": forecast
})

forecast_csv_path = os.path.join(output_path, "forecast_data.csv")
forecast_df.to_csv(forecast_csv_path)

# ---------------------------
# 6. Plot Graph
# ---------------------------
plt.figure()

monthly_sales.plot(label="Actual Sales")
forecast.plot(label="Forecast (Moving Avg)")

plt.legend()
plt.title("Demand Forecasting (Monthly Sales)")

# save plot
plot_path = os.path.join(output_path, "demand_forecast.png")
plt.savefig(plot_path)
plt.close()

# ---------------------------
# 7. Final Output
# ---------------------------
print(" Demand Forecasting Completed!")
print("Forecast CSV saved at:", forecast_csv_path)
print(" Plot saved at:", plot_path)