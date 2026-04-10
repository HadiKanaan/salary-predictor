from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI

try:
    from api.routers.predict import router as predict_router
    from api.routers.text import router as text_router
    from api.routers.train import router as train_router
    from api.services.model_service import ModelService
    from api.services.text_analysis_service import TextAnalysisService
    from api.services.training_service import TrainingService
except ModuleNotFoundError:
    from routers.predict import router as predict_router
    from routers.text import router as text_router
    from routers.train import router as train_router
    from services.model_service import ModelService
    from services.text_analysis_service import TextAnalysisService
    from services.training_service import TrainingService


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(raw_path: str) -> str:
    """Resolves a configured path against the project root when needed."""
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return str(candidate)
    return str(PROJECT_ROOT / candidate)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initializes and stores application services during startup."""
    load_dotenv(override=True)
    model_path = resolve_project_path(os.getenv("MODEL_PATH", "model/salary_model.pkl"))
    encoders_path = resolve_project_path(os.getenv("ENCODERS_PATH", "model/encoders.pkl"))
    training_data_path = resolve_project_path(
        os.getenv("TRAINING_DATA_PATH", "data/cleaned/ds_salaries_clean.csv")
    )
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    ollama_model = os.getenv("OLLAMA_MODEL")

    if not ollama_base_url or not ollama_model:
        raise RuntimeError("OLLAMA_BASE_URL and OLLAMA_MODEL must be set in .env.")

    model_service = ModelService(model_path=model_path, encoders_path=encoders_path)
    model_service.load()
    app.state.model_service = model_service

    training_service = TrainingService(data_path=training_data_path, model_path=model_path)
    app.state.training_service = training_service

    text_analysis_service = TextAnalysisService(
        base_url=ollama_base_url,
        model=ollama_model,
    )
    app.state.text_analysis_service = text_analysis_service

    yield


app = FastAPI(title="Salary Predictor API", lifespan=lifespan)
app.include_router(predict_router)
app.include_router(train_router)
app.include_router(text_router)