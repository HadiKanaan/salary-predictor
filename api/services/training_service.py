from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


class TrainingService:
    """Trains and persists the salary prediction model."""

    def __init__(self, data_path: str, model_path: str) -> None:
        self.data_path = data_path
        self.model_path = model_path

    def train(self) -> dict[str, Any]:
        """Trains the model and returns training metrics."""
        data_file = Path(self.data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Training data not found at '{self.data_path}'.")

        df = pd.read_csv(data_file)

        X = df.drop("salary_in_usd", axis=1)
        y = df["salary_in_usd"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            random_state=42,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        model_file = Path(self.model_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        with model_file.open("wb") as f:
            pickle.dump(model, f)

        return {
            "mae": float(mae),
            "r2": float(r2),
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "model_path": str(model_file),
        }
