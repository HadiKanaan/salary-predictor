from pydantic import BaseModel


class PredictionInputs(BaseModel):
    work_year: int
    experience_level: str
    employment_type: str
    job_title: str
    employee_residence: str
    remote_ratio: int
    company_location: str
    company_size: str


class PredictionResponse(BaseModel):
    predicted_salary_usd: float
    inputs: PredictionInputs
