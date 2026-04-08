import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle, os

df = pd.read_csv("data/raw/ds_salaries.csv")

# Drop columns you don't need
df = df.drop(columns=["salary", "salary_currency", "Unnamed: 0"], errors="ignore")

# Drop nulls
df = df.dropna()

# Columns to encode
cat_cols = ["experience_level", "employment_type", "job_title",
            "employee_residence", "company_location", "company_size"]

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save cleaned data
df.to_csv("data/cleaned/ds_salaries_clean.csv", index=False)

# Save encoders
os.makedirs("model", exist_ok=True)
with open("model/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Cleaning done.")
print(df.head())
print(df.dtypes)