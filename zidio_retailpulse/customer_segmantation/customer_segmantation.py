import pandas as pd
import datetime as dt
import os

# ---------------------------
# 1. Fix Paths
# ---------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "..", "DATASET", "feature_engineered_data.csv")
output_path = os.path.join(base_path, "outputs")

# create output folder
os.makedirs(output_path, exist_ok=True)

# ---------------------------
# 2. Load Data
# ---------------------------
df = pd.read_csv(data_path)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ---------------------------
# 3. Reference Date
# ---------------------------
today_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# ---------------------------
# 4. Create RFM Table
# ---------------------------
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (today_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalAmount': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# ---------------------------
# 5. RFM Scoring
# ---------------------------
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

rfm['RFM_Score'] = (
    rfm['R_Score'].astype(str) +
    rfm['F_Score'].astype(str) +
    rfm['M_Score'].astype(str)
)

# ---------------------------
# 6. Segmentation
# ---------------------------
def segment(row):
    if row['RFM_Score'] == '555':
        return 'Best Customers'
    elif int(row['F_Score']) >= 4:
        return 'Loyal Customers'
    elif int(row['R_Score']) >= 4:
        return 'Recent Customers'
    elif int(row['R_Score']) <= 2:
        return 'At Risk'
    else:
        return 'Regular Customers'

rfm['Segment'] = rfm.apply(segment, axis=1)

# ---------------------------
# 7. Save Files
# ---------------------------
rfm_file = os.path.join(base_path, "..", "DATASET", "rfm_analysis.csv")
rfm.to_csv(rfm_file)

# insights
segment_summary = rfm['Segment'].value_counts()

insight_file = os.path.join(output_path, "insights.txt")
with open(insight_file, "w") as f:
    f.write("RFM CUSTOMER SEGMENTATION\n\n")
    f.write(str(segment_summary))

print(" RFM DONE!")
print("Saved at:", rfm_file)
print(" Insights at:", insight_file)