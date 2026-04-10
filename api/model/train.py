import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/cleaned/ds_salaries_clean.csv")

X = df.drop("salary_in_usd", axis=1)
y = df["salary_in_usd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE:  ${mae:,.2f}")
print(f"R2:   {r2:.4f}")  # closer to 1.0 is better

# Feature importances
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importances:")
print(importances)

# Save model
with open("model/salary_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to model/salary_model.pkl")
print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")
print(f"MAE:              ${mae:,.2f}")
print(f"R2:               {r2:.4f}")
print(f"\nBaseline MAE (always predicting mean): ${(y_test - y_test.mean()).abs().mean():,.2f}")