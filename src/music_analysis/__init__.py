"""Phrase-oriented music analysis helpers for token sequences."""

from .phrase_analysis import (
    BarInfo,
    BoundaryScore,
    PhraseAnalysis,
    PhraseAnalysisConfig,
    PhraseSpan,
    PhraseWindowPolicy,
    SampledWindow,
    analyze_phrase_candidates,
    extract_phrase,
    sample_phrase_window,
)

__all__ = [
    "BarInfo",
    "BoundaryScore",
    "PhraseAnalysis",
    "PhraseAnalysisConfig",
    "PhraseSpan",
    "PhraseWindowPolicy",
    "SampledWindow",
    "analyze_phrase_candidates",
    "extract_phrase",
    "sample_phrase_window",
]
