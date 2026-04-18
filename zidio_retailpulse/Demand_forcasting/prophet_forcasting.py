import pandas as pd
import os
from prophet import Prophet
import matplotlib.pyplot as plt

# ---------------------------
# Paths
# ---------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "..", "DATASET", "feature_engineered_data.csv")
output_path = os.path.join(base_path, "outputs")

os.makedirs(output_path, exist_ok=True)

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv(data_path)

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# monthly data
df = df.groupby('InvoiceDate')['TotalAmount'].sum().reset_index()

df = df.rename(columns={"InvoiceDate": "ds", "TotalAmount": "y"})

# ---------------------------
# Prophet Model
# ---------------------------
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=3, freq='M')
forecast = model.predict(future)

# ---------------------------
# Plot
# ---------------------------
fig = model.plot(forecast)

plot_path = os.path.join(output_path, "prophet_forecast.png")
fig.savefig(plot_path)

# ---------------------------
# Save CSV
# ---------------------------
forecast[['ds', 'yhat']].to_csv(os.path.join(output_path, "prophet_forecast.csv"), index=False)

print(" Prophet Forecast ")