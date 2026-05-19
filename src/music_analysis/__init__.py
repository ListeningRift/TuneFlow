"""Music analysis helpers for TuneFlow token sequences."""

from .key_analysis import (
    KeyAnalysisConfig,
    KeyFrame,
    KeySegment,
    KeyTimelineAnalysis,
    ModulationPoint,
    analyze_key_timeline,
)
from .phrase_analysis import (
    BarInfo,
    BoundaryScore,
    PhraseAnalysis,
    PhraseAnalysisConfig,
    PhraseBoundary,
    PhraseSpan,
    analyze_phrase_candidates,
)

__all__ = [
    "BarInfo",
    "BoundaryScore",
    "KeyAnalysisConfig",
    "KeyFrame",
    "KeySegment",
    "KeyTimelineAnalysis",
    "ModulationPoint",
    "PhraseAnalysis",
    "PhraseAnalysisConfig",
    "PhraseBoundary",
    "PhraseSpan",
    "analyze_key_timeline",
    "analyze_phrase_candidates",
]
