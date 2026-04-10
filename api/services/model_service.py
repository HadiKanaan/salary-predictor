from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
from fastapi import HTTPException


class ModelService:
    """Provides model loading, encoding, and prediction operations."""

    def __init__(self, model_path: str, encoders_path: str) -> None:
        self.model_path = model_path
        self.encoders_path = encoders_path
        self.model: Any | None = None
        self.encoders: dict[str, Any] = {}

    def load(self) -> None:
        """Loads the trained model and label encoders from disk."""
        model_file = Path(self.model_path)
        encoders_file = Path(self.encoders_path)

        if not model_file.exists():
            raise RuntimeError(f"Model file not found at '{self.model_path}'.")
        if not encoders_file.exists():
            raise RuntimeError(f"Encoders file not found at '{self.encoders_path}'.")

        with model_file.open("rb") as f:
            self.model = pickle.load(f)

        with encoders_file.open("rb") as f:
            self.encoders = pickle.load(f)

    def encode_value(self, column: str, value: str) -> int:
        """Encodes a categorical value using the fitted encoder for its column."""
        if column not in self.encoders:
            raise HTTPException(status_code=500, detail=f"Encoder for '{column}' is not loaded.")

        label_encoder = self.encoders[column]
        valid_options = list(label_encoder.classes_)
        if value not in valid_options:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown value '{value}' for field '{column}'. Valid options: {valid_options}",
            )

        return int(label_encoder.transform([value])[0])

    def predict_salary(
        self,
        work_year: int,
        experience_level: str,
        employment_type: str,
        job_title: str,
        employee_residence: str,
        remote_ratio: int,
        company_location: str,
        company_size: str,
    ) -> float:
        """Runs a salary prediction from raw API inputs."""
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded.")

        exp = self.encode_value("experience_level", experience_level)
        emp = self.encode_value("employment_type", employment_type)
        job = self.encode_value("job_title", job_title)
        res = self.encode_value("employee_residence", employee_residence)
        loc = self.encode_value("company_location", company_location)
        size = self.encode_value("company_size", company_size)

        feature_map = {
            "work_year": work_year,
            "experience_level": exp,
            "employment_type": emp,
            "job_title": job,
            "employee_residence": res,
            "remote_ratio": remote_ratio,
            "company_location": loc,
            "company_size": size,
        }

        # Keep prediction input in the same feature order used during training.
        feature_order = list(getattr(self.model, "feature_names_in_", feature_map.keys()))
        features = np.array([[feature_map[col] for col in feature_order]], dtype=float)
        prediction = self.model.predict(features)[0]
        return float(prediction)

    def get_options(self) -> dict[str, list[str]]:
        """Returns all valid categorical options from loaded encoders."""
        return {column: list(label_encoder.classes_) for column, label_encoder in self.encoders.items()}
