from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ApiClient:
    """Handles calls to the salary predictor API."""

    base_url: str
    timeout: float = 20.0

    def _url(self, path: str) -> str:
        """Builds a full URL for the provided API path."""
        return f"{self.base_url.rstrip('/')}{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Sends an HTTP request and returns parsed JSON with normalized errors."""
        timeout = kwargs.pop("timeout", self.timeout)
        try:
            response = requests.request(method, self._url(path), timeout=timeout, **kwargs)
        except requests.ReadTimeout as exc:
            raise RuntimeError(
                "API request timed out. The server may still be processing (for text analysis, Ollama can be slow)."
            ) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError("Unable to reach the API. Ensure FastAPI is running.") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text or "Unexpected response from API."}

        if response.status_code >= 400:
            detail = payload.get("detail", "Request failed.")
            raise RuntimeError(str(detail))

        return payload

    def health(self) -> dict[str, Any]:
        """Fetches health status from the API."""
        return self._request("GET", "/health")

    def options(self) -> dict[str, list[str]]:
        """Fetches categorical options from the API."""
        data = self._request("GET", "/options")
        return {k: list(v) for k, v in data.items()}

    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        """Requests a salary prediction from the API."""
        return self._request("GET", "/predict", params=params)

    def train(self) -> dict[str, Any]:
        """Triggers model retraining through the API."""
        return self._request("POST", "/train")

    def analyze_text(self, text: str, task: str) -> dict[str, Any]:
        """Requests text analysis from the API's Ollama endpoint."""
        return self._request(
            "POST",
            "/analyze-text",
            json={"text": text, "task": task},
            timeout=None,
        )
