from pydantic import BaseModel


class TrainResponse(BaseModel):
    """Represents the outcome of a model training run."""

    mae: float
    r2: float
    training_samples: int
    test_samples: int
    model_path: str
