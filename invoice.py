# Importing all lib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv("/content/InvoiceDataset.csv")

# finding and filling the missing values 
print(df.isna().sum())
numeric_cols = ["Vendor_ID", "Invoice_Amount", "Item_Count", "Tax", "Total", "Amount_per_item"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df["Payment_Term"] = df["Payment_Term"].fillna(df["Payment_Term"].mode()[0])
df["Invoice_Category"] = df["Invoice_Category"].fillna(df["Invoice_Category"].mode()[0])
df["Invoice_Date"] = df["Invoice_Date"].fillna(df["Invoice_Date"].mode()[0])

# 3. Encode categorical data
df_enc = df.copy()
df_enc["Payment_Term"] = df_enc["Payment_Term"].astype("category").cat.codes
df_enc["Invoice_Category"] = df_enc["Invoice_Category"].astype("category").cat.codes

features = ["Vendor_ID","Invoice_Amount","Item_Count","Tax","Total","Amount_per_item","Payment_Term","Invoice_Category"]
X = df_enc[features]


# 5. Feature scaling

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Clustering 
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df["Predicted_Cluster"] = clusters

silhouette = silhouette_score(X_scaled, clusters)
dbi = davies_bouldin_score(X_scaled, clusters)
chi = calinski_harabasz_score(X_scaled, clusters)

# mapping clusters
mapped_cluster = df.groupby("Predicted_Cluster")["is_anomaly"].mean().idxmax()
df["Predicted_Label"] = df["Predicted_Cluster"].apply(lambda x: 1 if x == mapped_cluster else 0)

# Metrics
cm = confusion_matrix(df["is_anomaly"], df["Predicted_Label"])
accuracy = accuracy_score(df["is_anomaly"], df["Predicted_Label"])
precision = precision_score(df["is_anomaly"], df["Predicted_Label"])
recall = recall_score(df["is_anomaly"], df["Predicted_Label"])
f1 = f1_score(df["is_anomaly"], df["Predicted_Label"])

# centeroids
centroids = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids)

distances = []
for i in range(len(X_scaled)):
    centroid = centroids[df["Predicted_Cluster"].iloc[i]]
    distances.append(norm(X_scaled[i] - centroid))
df["Distance_From_Centroid"] = distances

plt.figure(figsize=(8,5))
sns.histplot(df["Distance_From_Centroid"], bins=20, kde=True)
plt.title("Distance Distribution from Cluster Centroids")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Centroids (scaled):")
print(centroids)
print("Centroids (original):")
print(pd.DataFrame(centroids_original, columns=features))
print("Silhouette:", silhouette)
print("DBI:", dbi)
print("CHI:", chi)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

