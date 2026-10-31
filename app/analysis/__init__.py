"""Derin analiz katmanı (Vision LLM).

from app.analysis import AnalysisWorker, AnalysisMode, create_vlm_client

client = create_vlm_client(settings.vlm)
with AnalysisWorker(client, model=settings.vlm.model) as worker:
    worker.submit(frame, mode=AnalysisMode.DESCRIBE)
    result = worker.wait_for_result(timeout=180)
"""

from __future__ import annotations

from .prompts import AnalysisMode, build_prompt
from .vlm_client import (
    OllamaClient,
    OpenAICompatibleClient,
    UnavailableVLMClient,
    VLMClient,
    VLMError,
    VLMResponseError,
    VLMTimeoutError,
    VLMUnavailableError,
    create_vlm_client,
)
from .worker import AnalysisRequest, AnalysisResult, AnalysisWorker

__all__ = [
    "AnalysisMode",
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisWorker",
    "OllamaClient",
    "OpenAICompatibleClient",
    "UnavailableVLMClient",
    "VLMClient",
    "VLMError",
    "VLMResponseError",
    "VLMTimeoutError",
    "VLMUnavailableError",
    "build_prompt",
    "create_vlm_client",
]
