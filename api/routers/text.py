from fastapi import APIRouter, Depends, HTTPException, Request

try:
    from api.models.text_schemas import TextAnalysisRequest, TextAnalysisResponse
    from api.services.text_analysis_service import TextAnalysisService
    from api.services.supabase_service import get_supabase_client
except ModuleNotFoundError:
    from models.text_schemas import TextAnalysisRequest, TextAnalysisResponse
    from services.text_analysis_service import TextAnalysisService
    from services.supabase_service import get_supabase_client


router = APIRouter()


def get_text_service(request: Request) -> TextAnalysisService:
    """Returns the text analysis service from app state."""
    return request.app.state.text_analysis_service


@router.post("/analyze-text", response_model=TextAnalysisResponse)
def analyze_text(
    payload: TextAnalysisRequest,
    text_service: TextAnalysisService = Depends(get_text_service),
) -> TextAnalysisResponse:
    """Analyzes user text using the configured Ollama model and persists result to Supabase."""
    result = text_service.analyze(text=payload.text, task=payload.task)

    try:
        supabase = get_supabase_client()
        supabase.table("analyses").insert({
            "input_text": payload.text,
            "task": result.get("task"),
            "model": result.get("model"),
            "narrative_title": result.get("narrative_title"),
            "narrative": result.get("narrative"),
            "story_points": result.get("story_points"),
            "theme_scores": result.get("theme_scores"),
            "key_phrases": result.get("key_phrases"),
        }).execute()
    except Exception as e:
        print(f"[Supabase] Insert failed: {e}")

    return TextAnalysisResponse(**result)


@router.get("/analyses/history")
def analyses_history(limit: int = 20) -> dict:
    """Fetches the most recent text analyses from Supabase."""
    try:
        supabase = get_supabase_client()
        data = (
            supabase.table("analyses")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"records": data.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analyses history: {e}")