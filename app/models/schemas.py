"""
Pydantic models for the Bias Detection Engine
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ArticleInput(BaseModel):
    title: str
    content: str
    source: str
    url: Optional[str] = None

class BiasScore(BaseModel):
    ideological_stance: float
    factual_grounding: float
    framing_choices: float
    emotional_tone: float
    source_transparency: float

class HighlightedPhrase(BaseModel):
    text: str
    score_impact: Dict[str, float]
    explanation: str
    confidence: float

class BiasAnalysisOutput(BaseModel):
    article_id: str
    overall_score: float
    dimension_scores: BiasScore
    confidence_intervals: Dict[str, List[float]]  # Now arrays [min, max]
    highlighted_phrases: List[HighlightedPhrase]
    processing_time_ms: float
    model_used: str
    timestamp: str

class BatchAnalysisRequest(BaseModel):
    articles: List[ArticleInput]
    analysis_options: Optional[Dict[str, Any]] = None

class BatchAnalysisResponse(BaseModel):
    analyses: List[BiasAnalysisOutput]
    batch_id: str
    total_processing_time_ms: float
    timestamp: str

class NarrativeClusterOutput(BaseModel):
    cluster_id: int
    size: int
    dominant_themes: List[str]
    bias_profile: Dict[str, float]
    representative_phrases: List[str]
    articles: List[Dict[str, Any]]
    clustering_explanation: Optional[str] = None  # New: explanation of why this cluster exists

class NewsAnalysisRequest(BaseModel):
    topic: Optional[str] = None
    sources: Optional[str] = None
    max_articles: int = 15

class NewsAnalysisResponse(BaseModel):
    topic: str
    articles_analyzed: int
    processing_time_ms: float
    processing_time_per_article_ms: float
    narrative_clusters: List[NarrativeClusterOutput]
    cluster_visualizations: Optional[Dict[str, Any]]
    story_coverage_analysis: Optional[Dict[str, Any]]
    clustering_insights: Optional[Dict[str, Any]]  # New: transparency about clustering decisions
    clustering_recommendation: Optional[Dict[str, Any]] = None  # New: whether clustering was appropriate
    timestamp: str

class OutletComparisonRequest(BaseModel):
    """Request model for comparing how different outlets frame the same story"""
    articles: List[Dict[str, Any]]  # List of articles with title, content, source, and optional bias_scores
    comparison_focus: Optional[str] = None  # Optional focus area (e.g., "headlines", "sources", "tone")
    include_metrics: bool = True  # Whether to include quantitative comparison metrics

class OutletComparisonResponse(BaseModel):
    """Response model for cross-outlet comparison analysis"""
    comparison_analysis: str  # AI-generated analysis text
    comparison_metrics: Dict[str, Any]  # Quantitative metrics
    articles_compared: int
    outlets_analyzed: List[str]
    processing_time_ms: float
    model_used: str
    timestamp: str