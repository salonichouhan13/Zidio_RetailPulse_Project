import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA

# ---------------------------
# 1. Fix Paths
# ---------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "..", "DATASET", "feature_engineered_data.csv")
output_path = os.path.join(base_path, "outputs")

os.makedirs(output_path, exist_ok=True)

# ---------------------------
# 2. Load Data
# ---------------------------
df = pd.read_csv(data_path)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ---------------------------
# 3. Time Series (Monthly)
# ---------------------------
df.set_index('InvoiceDate', inplace=True)
monthly_sales = df['TotalAmount'].resample('M').sum()

# ---------------------------
# 4. Apply ARIMA Model
# ---------------------------
model = ARIMA(monthly_sales, order=(1,1,1))
model_fit = model.fit()

# ---------------------------
# 5. Forecast next 3 months
# ---------------------------
forecast = model_fit.forecast(steps=3)

# ---------------------------
# 6. Plot
# ---------------------------
plt.figure()

monthly_sales.plot(label="Actual Sales")
forecast.plot(label="Forecast (ARIMA)", color='red')

plt.legend()
plt.title("Demand Forecasting using ARIMA")

plot_path = os.path.join(output_path, "arima_forecast.png")
plt.savefig(plot_path)
plt.close()

# ---------------------------
# 7. Save Forecast Data
# ---------------------------
forecast_df = pd.DataFrame({
    "Forecast": forecast
})

forecast_csv = os.path.join(output_path, "arima_forecast.csv")
forecast_df.to_csv(forecast_csv)

# ---------------------------
# 8. Output
# ---------------------------
print(" Advanced Forecasting (ARIMA) ")
print(" Saved CSV:", forecast_csv)
print(" Saved Plot:", plot_path)