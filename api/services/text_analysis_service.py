from __future__ import annotations

import json
from dataclasses import dataclass

import requests
from fastapi import HTTPException


@dataclass
class TextAnalysisService:
    """Handles text analysis requests through Ollama."""

    base_url: str
    model: str

    def _build_prompt(self, text: str, task: str) -> str:
        """Builds a strict prompt for structured text analysis output."""
        return (
            "You are a storytelling-focused text analysis engine. "
            "Return strict JSON only with keys: narrative_title, narrative, story_points, "
            "theme_scores, key_phrases. "
            "story_points must be an array of concise bullet strings. "
            "theme_scores must be an array of 3 to 6 objects. "
            "Each object must be {\"theme\": string, \"score\": number}. "
            "score must be between 0 and 100. "
            "key_phrases must be an array of short strings. "
            f"Task: {task}. "
            f"Text: {text}"
        )

    def _parse_response(self, raw_text: str) -> dict[str, object]:
        """Parses Ollama output into narrative fields and chart-ready themes."""
        if not raw_text:
            return {
                "narrative_title": "No insight generated",
                "narrative": "No analysis returned.",
                "story_points": ["The model did not return analysable output."],
                "theme_scores": [{"theme": "Overall Signal", "score": 50.0}],
                "key_phrases": [],
            }

        try:
            payload = json.loads(raw_text)
            narrative_title = str(payload.get("narrative_title", "")).strip() or "Narrative Insight"
            narrative = str(payload.get("narrative", "")).strip() or "No analysis returned."

            story_points_raw = payload.get("story_points", [])
            if isinstance(story_points_raw, list):
                story_points = [str(item).strip() for item in story_points_raw if str(item).strip()]
            else:
                story_points = []

            if not story_points:
                story_points = [narrative]

            theme_scores_raw = payload.get("theme_scores", [])
            theme_scores: list[dict[str, object]] = []
            if isinstance(theme_scores_raw, list):
                for item in theme_scores_raw:
                    if not isinstance(item, dict):
                        continue
                    theme = str(item.get("theme", "")).strip()
                    try:
                        score = float(item.get("score", 0))
                    except (TypeError, ValueError):
                        continue
                    if not theme:
                        continue
                    score = max(0.0, min(100.0, score))
                    theme_scores.append({"theme": theme, "score": score})

            if not theme_scores:
                theme_scores = [{"theme": "Overall Signal", "score": 50.0}]

            phrases_raw = payload.get("key_phrases", [])
            if isinstance(phrases_raw, list):
                key_phrases = [str(item).strip() for item in phrases_raw if str(item).strip()]
            else:
                key_phrases = []

            return {
                "narrative_title": narrative_title,
                "narrative": narrative,
                "story_points": story_points,
                "theme_scores": theme_scores,
                "key_phrases": key_phrases,
            }
        except json.JSONDecodeError:
            # Fallback for non-JSON outputs from smaller models.
            fallback_narrative = raw_text.strip() or "No analysis returned."
            return {
                "narrative_title": "Narrative Insight",
                "narrative": fallback_narrative,
                "story_points": [fallback_narrative],
                "theme_scores": [{"theme": "Overall Signal", "score": 50.0}],
                "key_phrases": [],
            }

    def analyze(self, text: str, task: str) -> dict[str, object]:
        """Runs a text analysis task via Ollama and returns structured output."""
        endpoint = f"{self.base_url.rstrip('/')}/api/generate"
        prompt = self._build_prompt(text=text, task=task)

        try:
            response = requests.post(
                endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=503,
                detail="Unable to reach Ollama. Ensure Ollama is running and reachable.",
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="Invalid response received from Ollama.") from exc

        if response.status_code >= 400:
            detail = payload.get("error", payload.get("detail", "Ollama request failed."))
            raise HTTPException(status_code=502, detail=f"Ollama error: {detail}")

        raw_text = str(payload.get("response", ""))
        parsed = self._parse_response(raw_text)

        return {
            "task": task,
            "model": self.model,
            "narrative_title": str(parsed["narrative_title"]),
            "narrative": str(parsed["narrative"]),
            "story_points": list(parsed["story_points"]),
            "theme_scores": list(parsed["theme_scores"]),
            # Keep this field for backward compatibility with existing UI.
            "analysis": str(parsed["narrative"]),
            "key_phrases": list(parsed["key_phrases"]),
        }
