
# INVOICE ANOMALY DETECTOR (K-Means Clustering)
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, accuracy_score, silhouette_score,
    confusion_matrix
)
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv(r"C:\Users\91630\Desktop\invoice_anomaly_dataset.csv")

print(" Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(df.head())


# 2. Data Preprocessing
df["Invoice_Date"] = pd.to_datetime(df["Invoice_Date"], errors='coerce')
df["Days_Since_Start"] = (df["Invoice_Date"] - df["Invoice_Date"].min()).dt.days.fillna(0)

# Map payment terms (e.g., Net30 -> 30)
term_map = {"Net15": 15, "Net30": 30, "Net45": 45, "Net60": 60}
df["Payment_Term_Days"] = df["Payment_Term"].map(term_map).fillna(30)

# Encode category
cat_map = {c: i for i, c in enumerate(df["Invoice_Category"].unique())}
df["Category_Code"] = df["Invoice_Category"].map(cat_map).fillna(-1)

# Fill missing numeric fields
print("\nMissing values before filling:")
print(df.isnull().sum())
num_cols = ["Invoice_Amount", "Tax", "Total", "Item_Count", "Amount_per_Item"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())


# 3. Feature Engineering
vendor_stats = df.groupby("Vendor_ID")["Invoice_Amount"].agg(["mean", "std", "count"]).rename(
    columns={"mean": "Vendor_Avg", "std": "Vendor_Std", "count": "Vendor_Count"})
df = df.merge(vendor_stats, on="Vendor_ID", how="left")

df["Deviation_From_Vendor_Avg"] = abs(df["Invoice_Amount"] - df["Vendor_Avg"])
df["Tax_Ratio"] = df["Tax"] / (df["Total"] + 1e-9)


# 4. Select Features
features = ["Invoice_Amount", "Tax", "Total", "Item_Count", "Amount_per_Item",
            "Days_Since_Start", "Payment_Term_Days", "Category_Code",
            "Vendor_Avg", "Vendor_Std", "Vendor_Count",
            "Deviation_From_Vendor_Avg", "Tax_Ratio"]

X = df[features].fillna(0).astype(float)
y_true = df["is_anomaly"].astype(int)

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Find Optimal K

best_k = 0
best_score = -1
for k in range(3, 10):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    if score > best_score:
        best_k = k
        best_score = score

print(f"\n Best K = {best_k} (Silhouette Score = {best_score:.3f})")


# 6. Train Final K-Means Model

kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X_scaled)

# Compute distance from cluster centers
distances = cdist(X_scaled, kmeans.cluster_centers_, 'euclidean')
min_distances = np.min(distances, axis=1)

# Dynamic anomaly threshold
threshold = np.mean(min_distances) + 2 * np.std(min_distances)
df["kmeans_pred"] = (min_distances > threshold).astype(int)
df["kmeans_distance"] = min_distances

print(f"\nAnomaly threshold: {threshold:.3f}")
print(f"Total anomalies detected: {df['kmeans_pred'].sum()}")


# 7. Model Evaluation

def evaluate_model(y_true, y_pred, score):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, score)
    accuracy = accuracy_score(y_true, y_pred)
    print("\n Model Performance (K-Means Clustering):")
    print(f"Accuracy={accuracy:.3f}  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
    return accuracy, precision, recall, f1, auc

accuracy, precision, recall, f1, auc = evaluate_model(y_true, df["kmeans_pred"], df["kmeans_distance"])

# ------------------------------
# 8. Results Table
# ------------------------------
results_df = pd.DataFrame({
    "Model": ["K-Means Clustering"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1-Score": [f1],
    "AUC": [auc]
})

print("\n================= MODEL RESULTS =================")
print(results_df.to_string(index=False))


# Confusion Matrix
cm = confusion_matrix(y_true, df["kmeans_pred"])
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - K-Means")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Distance Distribution
plt.figure(figsize=(7,4))
sns.histplot(df["kmeans_distance"], bins=50, color='skyblue', kde=True)
plt.axvline(df["kmeans_distance"].mean(), color='red', linestyle='--', label='Mean Distance')
plt.axvline(df["kmeans_distance"].mean() + 2*df["kmeans_distance"].std(),
            color='orange', linestyle='--', label='Threshold')
plt.title("Distribution of K-Means Distances")
plt.xlabel("Distance to Cluster Center")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# ------------------------------
# 🔍 10. Predict New Invoice & Explain WHY Anomaly
# ------------------------------
def predict_invoice_anomaly(invoice_amount, tax, total, item_count, amount_per_item, payment_term_days=30, category_code=0, vendor_avg=0, vendor_std=0, vendor_count=0):
    """
    Predict if a given invoice is an anomaly or normal, and explain why.
    """
    # Create new data
    new_data = pd.DataFrame([[
        invoice_amount, tax, total, item_count, amount_per_item,
        0, payment_term_days, category_code, vendor_avg, vendor_std, vendor_count,
        abs(invoice_amount - vendor_avg), tax / (total + 1e-9)
    ]], columns=features)

    # Scale input
    new_scaled = scaler.transform(new_data)

    # Compute distance to cluster centers
    distances = cdist(new_scaled, kmeans.cluster_centers_, 'euclidean')
    min_distance = np.min(distances)
    is_anomaly = int(min_distance > threshold)

    print("\n New Invoice Prediction:")
    print(f"Distance from nearest cluster: {min_distance:.3f}")
    print(f"Threshold for anomaly: {threshold:.3f}")

    # Reasoning for anomaly
    if is_anomaly:
        reasons = []
        if invoice_amount > df["Invoice_Amount"].mean() + 2 * df["Invoice_Amount"].std():
            reasons.append("Unusually high invoice amount")
        if item_count < df["Item_Count"].mean() - 1.5 * df["Item_Count"].std():
            reasons.append("Very few items in invoice")
        if (tax / (total + 1e-9)) > df["Tax_Ratio"].mean() + 2 * df["Tax_Ratio"].std():
            reasons.append("Abnormally high tax ratio")
        if abs(invoice_amount - vendor_avg) > df["Deviation_From_Vendor_Avg"].mean() + 2 * df["Deviation_From_Vendor_Avg"].std():
            reasons.append("Large deviation from vendor's usual invoice amount")

        print(" Result: This invoice is an ANOMALY.")
        if reasons:
            print("Possible Reasons:")
            for r in reasons:
                print(f" - {r}")
        else:
            print(" - Does not fit into any known cluster pattern.")
    else:
        print(" Result: This invoice is NORMAL (fits known patterns).")

    return is_anomaly


# Example Test Prediction
predict_invoice_anomaly(
    invoice_amount=9200,  # large amount
    tax=800,
    total=10000,
    item_count=2,  # very few items
    amount_per_item=4600,
    payment_term_days=30,
    vendor_avg=3500,
    vendor_std=500,
    vendor_count=20
)
