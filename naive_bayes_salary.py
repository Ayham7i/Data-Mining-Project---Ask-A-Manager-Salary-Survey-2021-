# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------
# Load dataset (replace with your file path)
try:
    df = pd.read_excel("Ask A Manager Salary Survey 2021 (Responses).xlsx", sheet_name="Form Responses 1")
except ImportError:
    df = pd.read_csv("Ask A Manager Salary Survey 2021 (Responses).csv")

# Convert currency to USD and handle missing rates
exchange_rates = {"gbp": 1.37, "cad": 0.79, "usd": 1.0, "eur": 1.18}
df["annual_salary_usd"] = df.apply(
    lambda x: x["annual salary"] * exchange_rates.get(x["currency"].lower(), 1), 
    axis=1
)

# Bin salaries into categories (target variable)
bins = [0, 50000, 80000, 120000, float('inf')]
labels = ["<50k", "50k-80k", "80k-120k", "120k+"]
df["salary_bin"] = pd.cut(df["annual_salary_usd"], bins=bins, labels=labels)

# Bin experience
experience_mapping = {
    "1 year or less": "Entry",
    "2 - 4 years": "Junior",
    "5-7 years": "Mid",
    "8 - 10 years": "Senior",
    "11 - 20 years": "Expert",
    "21 - 30 years": "Veteran",
    "31 - 40 years": "Veteran",
    "41 years or more": "Veteran"
}
df["experience_bin"] = df["overall years of professional experience"].map(experience_mapping)

# Clean education and industry
df["education"] = df["highest level of education completed"].str.lower().str.replace("'", "")
df["industry"] = df["industry"].str.lower()

# Remove outliers
df = df[df["annual_salary_usd"] > 1000]

# Select relevant features and target
features = ["industry", "education", "experience_bin", "gender"]
target = "salary_bin"
df = df[features + [target]].dropna()

# ---------------------------
# 2. Encode Categorical Features
# ---------------------------
# Label encode categorical variables
encoders = {}
for col in features + [target]:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Split into features (X) and target (y)
X = df[features]
y = df[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------
# 3. Train Naïve Bayes Classifier
# ---------------------------
# Initialize and train the model
model = CategoricalNB()  # Works with categorical integer-encoded features
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# ---------------------------
# 4. Evaluate Performance
# ---------------------------
# Decode labels for readability
salary_labels = encoders[target].classes_

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=salary_labels))