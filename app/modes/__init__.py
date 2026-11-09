"""Çift modlu çalışma biçimleri.

* :class:`AutopilotMode` — sürekli, ucuz tarama (engel + banknot).
* :class:`DeepAnalysisMode` — istek üzerine kare dondurma + VLM + seslendirme.
"""

from __future__ import annotations

from .autopilot import BANKNOTE_LABELS, AutopilotMode, AutopilotStats
from .deep import DeepAnalysisMode, DeepAnalysisState, ExpiryInfo, parse_expiry

__all__ = [
    "BANKNOTE_LABELS",
    "AutopilotMode",
    "AutopilotStats",
    "DeepAnalysisMode",
    "DeepAnalysisState",
    "ExpiryInfo",
    "parse_expiry",
]
