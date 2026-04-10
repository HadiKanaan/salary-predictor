from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from supabase import Client, create_client


@dataclass
class DashboardSupabaseClient:
    """Lightweight Supabase wrapper for Streamlit read/write flows."""

    url: str
    key: str

    def _client(self) -> Client:
        return create_client(self.url, self.key)

    def insert_prediction(self, payload: dict[str, Any]) -> None:
        """Inserts a prediction row into the predictions table."""
        self._client().table("predictions").insert(payload).execute()

    def fetch_predictions(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetches recent prediction rows ordered by creation time."""
        response = (
            self._client()
            .table("predictions")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return list(response.data or [])

    def insert_analysis(self, payload: dict[str, Any]) -> None:
        """Inserts a text analysis row into the analyses table."""
        self._client().table("analyses").insert(payload).execute()

    def fetch_analyses(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetches recent text analysis rows ordered by creation time."""
        response = (
            self._client()
            .table("analyses")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return list(response.data or [])
