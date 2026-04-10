from __future__ import annotations

from pydantic import BaseModel, Field


class TextAnalysisRequest(BaseModel):
    """Defines the payload for text analysis requests."""

    text: str = Field(..., min_length=1, max_length=5000)
    task: str = Field(default="general", min_length=2, max_length=50)


class ThemeScore(BaseModel):
    """Defines a scored narrative theme for charting."""

    theme: str
    score: float


class TextAnalysisResponse(BaseModel):
    """Defines the structured response for text analysis."""

    task: str
    model: str
    narrative_title: str
    narrative: str
    story_points: list[str]
    theme_scores: list[ThemeScore]
    analysis: str
    key_phrases: list[str]
