import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

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

# ---------------------------
# RFM
# ---------------------------
snapshot_date = df['InvoiceDate'].max()

rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'Invoice': 'count',
    'TotalAmount': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# ---------------------------
# Churn Label
# ---------------------------
rfm['Churn'] = 0

rfm.loc[
    (rfm['Recency'] > 90) &
    (rfm['Frequency'] < 5),
    'Churn'
] = 1

# ---------------------------
# ML
# ---------------------------
X = rfm[['Recency', 'Frequency', 'Monetary']]
y = rfm['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------------------
# Accuracy
# ---------------------------
acc = accuracy_score(y_test, y_pred)

# ---------------------------
# Save Results
# ---------------------------
with open(os.path.join(output_path, "model_results.txt"), "w") as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write(classification_report(y_test, y_pred))

# ---------------------------
# CONFUSION MATRIX GRAPH
# ---------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")

plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
plt.close()

# ---------------------------
# FEATURE IMPORTANCE
# ---------------------------
importance = model.coef_[0]

plt.figure()
plt.bar(['Recency', 'Frequency', 'Monetary'], importance)
plt.title("Feature Importance")

plt.savefig(os.path.join(output_path, "feature_importance.png"))
plt.close()

# ---------------------------
# Output
# ---------------------------
print(" ML Churn Model with Graph Done!")
print("Accuracy:", acc)