"""TuneFlow token 序列的音乐分析工具。"""

from .key_analysis import (
    KeyAnalysisConfig,
    KeyFrame,
    KeySegment,
    KeyTimelineAnalysis,
    ModulationPoint,
    analyze_key_timeline,
)
from .phrase_analysis import (
    AnalysisAnchor,
    BarInfo,
    BoundaryFeature,
    BoundaryScore,
    HierarchicalBoundaryScore,
    NoteInfo,
    PhraseAnalysis,
    PhraseAnalysisConfig,
    PhraseBoundary,
    PhraseSpan,
    analyze_phrase_candidates,
)

__all__ = [
    "AnalysisAnchor",
    "BarInfo",
    "BoundaryFeature",
    "BoundaryScore",
    "HierarchicalBoundaryScore",
    "KeyAnalysisConfig",
    "KeyFrame",
    "KeySegment",
    "KeyTimelineAnalysis",
    "ModulationPoint",
    "NoteInfo",
    "PhraseAnalysis",
    "PhraseAnalysisConfig",
    "PhraseBoundary",
    "PhraseSpan",
    "analyze_key_timeline",
    "analyze_phrase_candidates",
]
