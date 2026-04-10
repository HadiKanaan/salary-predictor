from fastapi import APIRouter, Depends, Request

try:
    from api.models.training_schemas import TrainResponse
    from api.services.model_service import ModelService
    from api.services.training_service import TrainingService
except ModuleNotFoundError:
    from models.training_schemas import TrainResponse
    from services.model_service import ModelService
    from services.training_service import TrainingService


router = APIRouter()


def get_training_service(request: Request) -> TrainingService:
    """Returns the application training service from app state."""
    return request.app.state.training_service


def get_model_service(request: Request) -> ModelService:
    """Returns the application model service from app state."""
    return request.app.state.model_service


@router.post("/train", response_model=TrainResponse)
def train(
    training_service: TrainingService = Depends(get_training_service),
    model_service: ModelService = Depends(get_model_service),
) -> TrainResponse:
    """Trains the model and persists the updated artifact."""
    result = training_service.train()
    model_service.load()
    return TrainResponse(**result)
