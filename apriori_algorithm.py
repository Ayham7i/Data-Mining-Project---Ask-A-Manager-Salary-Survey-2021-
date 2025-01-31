import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------
# Load dataset (Excel or CSV)
try:
    df = pd.read_excel("Ask A Manager Salary Survey 2021 (Responses).xlsx", sheet_name="Form Responses 1")
except ImportError:
    # Fallback to CSV if openpyxl is not installed
    df = pd.read_csv("Ask A Manager Salary Survey 2021 (Responses).csv")

# Convert currency to USD
exchange_rates = {
    "gbp": 1.37,  # British Pound to USD (2021 rate)
    "cad": 0.79,  # Canadian Dollar to USD (2021 rate)
    "usd": 1.0,   # US Dollar (no conversion needed)
    "eur": 1.18   # Euro to USD (2021 rate)
}

# Handle missing currencies by defaulting to 1 (no conversion)
df["annual_salary_usd"] = df.apply(
    lambda x: x["annual salary"] * exchange_rates.get(x["currency"].lower(), 1), 
    axis=1
)

# Bin salaries
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

# Clean education
df["education"] = df["highest level of education completed"].str.lower().str.replace("'", "")

# Remove salary outliers (e.g., $58 entry)
df = df[df["annual_salary_usd"] > 1000]

# ---------------------------
# 2. Convert to Transactions
# ---------------------------
transactions = []
for _, row in df.iterrows():
    transaction = [
        f"industry={row['industry'].strip().lower()}",
        f"job={row['job title'].strip().lower()}",
        f"salary={row['salary_bin']}",
        f"education={row['education']}",
        f"exp={row['experience_bin']}",
        f"gender={row['gender'].strip().lower()}"
    ]
    transactions.append(transaction)

# ---------------------------
# 3. Run Apriori Algorithm
# ---------------------------
# One-hot encode transactions
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Find frequent itemsets (lower min_support for small dataset)
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
rules = rules.sort_values("lift", ascending=False)

# ---------------------------
# 4. Analyze Results
# ---------------------------
print("\nTop 10 Association Rules:")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

# Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.scatter(rules["support"], rules["confidence"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Association Rules: Support vs Confidence")
plt.show()