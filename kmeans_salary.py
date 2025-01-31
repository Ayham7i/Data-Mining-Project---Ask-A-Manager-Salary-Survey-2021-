# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------
# Load dataset
try:
    df = pd.read_excel("Ask A Manager Salary Survey 2021 (Responses).xlsx", sheet_name="Form Responses 1")
except ImportError:
    df = pd.read_csv("Ask A Manager Salary Survey 2021 (Responses).csv")

# Rename columns for simplicity
df = df.rename(columns={
    "highest level of education completed": "education",
    "overall years of professional experience": "experience"
})

# Convert currency to USD
exchange_rates = {"gbp": 1.37, "cad": 0.79, "usd": 1.0, "eur": 1.18}
df["annual_salary_usd"] = df.apply(
    lambda x: x["annual salary"] * exchange_rates.get(x["currency"].lower(), 1), 
    axis=1
)

# Select relevant features
features = ["industry", "education", "experience", "annual_salary_usd"]
df = df[features].dropna()

# Clean categorical features
df["industry"] = df["industry"].str.lower().str.strip()
df["education"] = df["education"].str.lower().str.replace("'", "")

# ---------------------------
# 2. Preprocess Data for Clustering
# ---------------------------
# Define categorical and numerical features
categorical_features = ["industry", "education"]
numerical_features = ["annual_salary_usd", "experience"]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)

# Apply preprocessing
X = preprocessor.fit_transform(df)

# ---------------------------
# 3. Determine Optimal Clusters (Elbow Method)
# ---------------------------
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker="o", linestyle="--")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.title("Elbow Method for Optimal K")
plt.show()

# ---------------------------
# 4. Apply K-means Clustering
# ---------------------------
# Choose K based on the elbow curve (e.g., K=4)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)

# Add clusters to dataframe
df["cluster"] = clusters

# ---------------------------
# 5. Visualize Clusters (PCA)
# ---------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.6)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-means Clusters (K=4)")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()

# ---------------------------
# 6. Interpret Clusters
# ---------------------------
cluster_summary = df.groupby("cluster").agg({
    "annual_salary_usd": ["mean", "std"],
    "experience": ["mean", "std"],
    "industry": lambda x: x.mode()[0],
    "education": lambda x: x.mode()[0]
}).reset_index()

print("\nCluster Summary:")
print(cluster_summary)