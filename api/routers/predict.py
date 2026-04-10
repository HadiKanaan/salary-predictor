from fastapi import APIRouter, Depends, Query, Request, HTTPException

try:
    from api.models.schemas import PredictionInputs, PredictionResponse
    from api.services.model_service import ModelService
    from api.services.supabase_service import get_supabase_client
except ModuleNotFoundError:
    from models.schemas import PredictionInputs, PredictionResponse
    from services.model_service import ModelService
    from services.supabase_service import get_supabase_client


router = APIRouter()


def get_model_service(request: Request) -> ModelService:
    """Returns the application model service from app state."""
    return request.app.state.model_service


@router.get("/predict", response_model=PredictionResponse)
def predict(
    work_year: int = Query(2023, description="Year e.g. 2020"),
    experience_level: str = Query(..., description="EN, MI, SE or EX"),
    employment_type: str = Query(..., description="FT, PT, CT or FL"),
    job_title: str = Query(..., description="e.g. Data Scientist"),
    employee_residence: str = Query(..., description="Country code e.g. US"),
    remote_ratio: int = Query(..., description="0, 50 or 100"),
    company_location: str = Query(..., description="Country code e.g. US"),
    company_size: str = Query(..., description="S, M or L"),
    model_service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Predicts salary in USD and persists the result to Supabase."""
    prediction = model_service.predict_salary(
        work_year=work_year,
        experience_level=experience_level,
        employment_type=employment_type,
        job_title=job_title,
        employee_residence=employee_residence,
        remote_ratio=remote_ratio,
        company_location=company_location,
        company_size=company_size,
    )

    rounded = round(prediction, 2)

    # Persist to Supabase — failure here should never break the prediction response
    try:
        supabase = get_supabase_client()
        supabase.table("predictions").insert({
            "work_year": work_year,
            "experience_level": experience_level,
            "employment_type": employment_type,
            "job_title": job_title,
            "employee_residence": employee_residence,
            "remote_ratio": remote_ratio,
            "company_location": company_location,
            "company_size": company_size,
            "predicted_salary_usd": rounded,
        }).execute()
    except Exception as e:
        print(f"[Supabase] Insert failed: {e}")

    return PredictionResponse(
        predicted_salary_usd=rounded,
        inputs=PredictionInputs(
            work_year=work_year,
            experience_level=experience_level,
            employment_type=employment_type,
            job_title=job_title,
            employee_residence=employee_residence,
            remote_ratio=remote_ratio,
            company_location=company_location,
            company_size=company_size,
        ),
    )


@router.get("/history")
def history(limit: int = Query(20, description="Number of records to return")) -> dict:
    """Fetches the most recent predictions from Supabase."""
    try:
        supabase = get_supabase_client()
        data = (
            supabase.table("predictions")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"records": data.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {e}")


@router.get("/options")
def options(model_service: ModelService = Depends(get_model_service)) -> dict[str, list[str]]:
    """Returns valid category values for each encoded field."""
    return model_service.get_options()


@router.get("/health")
def health() -> dict[str, str]:
    """Returns the API health status."""
    return {"status": "ok"}