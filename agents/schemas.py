"""
Pydantic Schemas — FEAT-004: Unified Analysis Report
------------------------------------------------------
Canonical output contract for both Quick and Deep analysis modes.
Every run produces a UnifiedAnalysisReport regardless of mode.
Old SentimentReport kept as alias for backwards compatibility during migration.
"""

import uuid
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ============================================================================
# META
# ============================================================================

class DataFreshness(BaseModel):
    earliest_post: Optional[str] = None
    latest_post: Optional[str] = None


class UnifiedReportMeta(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    mode: str = "deep"  # "quick" | "deep"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    documents_analyzed: int = 0
    retrieval_strategy: str = "hybrid"
    model_used: str = "gemini-2.0-flash"
    data_freshness: DataFreshness = Field(default_factory=DataFreshness)


# ============================================================================
# VERDICT
# ============================================================================

class SentimentVerdict(BaseModel):
    overall_sentiment: str = "Mixed"  # Positive | Negative | Mixed | Neutral
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    net_sentiment_score: int = 0       # computed: positive_pct - negative_pct
    one_line_summary: str = ""


# ============================================================================
# SIGNAL BLOCKS (Positive & Negative)
# ============================================================================

class RepresentativeQuote(BaseModel):
    text: str = ""
    score: int = 0
    source_post: Optional[str] = None


class SignalTheme(BaseModel):
    theme: str
    frequency: str = "Common"          # Universal | Common | Rare
    evidence_count: int = 0
    description: str = ""
    representative_quote: RepresentativeQuote = Field(default_factory=RepresentativeQuote)


class SignalQuote(BaseModel):
    text: str
    score: int = 0
    context: str = ""
    source_post: Optional[str] = None


class SignalBlock(BaseModel):
    headline: str = ""
    percentage: float = 0.0
    top_themes: List[SignalTheme] = Field(default_factory=list)
    praise_quotes: List[SignalQuote] = Field(default_factory=list)       # positive block
    criticism_quotes: List[SignalQuote] = Field(default_factory=list)    # negative block


# ============================================================================
# SENTIMENT DISTRIBUTION
# ============================================================================

class EmotionMap(BaseModel):
    anger: float = 0
    frustration: float = 0
    hope: float = 0
    satisfaction: float = 0
    disappointment: float = 0
    excitement: float = 0
    sarcasm: float = 0
    resignation: float = 0


class SentimentDistribution(BaseModel):
    positive_pct: float = Field(default=0.0, ge=0, le=100)
    negative_pct: float = Field(default=0.0, ge=0, le=100)
    neutral_pct: float = Field(default=0.0, ge=0, le=100)
    dominant_emotions: List[str] = Field(default_factory=list)
    emotion_map: EmotionMap = Field(default_factory=EmotionMap)
    sarcasm_detected: bool = False
    controversy_score: float = Field(default=0.0, ge=0.0, le=1.0)


# ============================================================================
# ENTITIES & SIGNALS
# ============================================================================

class KeyEntity(BaseModel):
    name: str
    type: str = "Other"   # Product | Person | Feature | Event | Other
    mention_count: int = 0
    net_sentiment: str = "Mixed"


class CompetitiveSignals(BaseModel):
    mentions_competitors: bool = False
    competitors_mentioned: List[str] = Field(default_factory=list)
    comparison_sentiment: str = "N/A"   # Favourable | Unfavourable | Neutral | N/A


class TrendIndicators(BaseModel):
    sentiment_trajectory: str = "Insufficient Data"  # Improving|Declining|Stable|Insufficient Data
    urgent_concerns: List[str] = Field(default_factory=list)
    emerging_positives: List[str] = Field(default_factory=list)


# ============================================================================
# ACTIONABLE INSIGHTS
# ============================================================================

class ActionableInsights(BaseModel):
    for_product_team: List[str] = Field(default_factory=list)
    for_marketing_team: List[str] = Field(default_factory=list)
    for_support_team: List[str] = Field(default_factory=list)


# ============================================================================
# MASTER REPORT — UnifiedAnalysisReport
# ============================================================================

class UnifiedAnalysisReport(BaseModel):
    """
    Canonical output for both Quick and Deep analysis modes.
    FEAT-004: All fields must always be present. Missing data uses empty defaults.
    """
    meta: UnifiedReportMeta
    verdict: SentimentVerdict = Field(default_factory=SentimentVerdict)
    executive_summary: str = ""
    positive_signals: SignalBlock = Field(default_factory=SignalBlock)
    negative_signals: SignalBlock = Field(default_factory=SignalBlock)
    sentiment_distribution: SentimentDistribution = Field(default_factory=SentimentDistribution)
    key_entities: List[KeyEntity] = Field(default_factory=list)
    competitive_signals: CompetitiveSignals = Field(default_factory=CompetitiveSignals)
    trend_indicators: TrendIndicators = Field(default_factory=TrendIndicators)
    actionable_insights: ActionableInsights = Field(default_factory=ActionableInsights)


# ============================================================================
# HELPERS
# ============================================================================

def make_fallback_report(query: str, mode: str, raw_text: str = "",
                         docs_analyzed: int = 0, strategy: str = "hybrid",
                         model: str = "unknown") -> dict:
    """
    Build a valid fallback UnifiedAnalysisReport when LLM JSON parsing fails.
    Guarantees all keys are present so the frontend never gets a KeyError.
    """
    report = UnifiedAnalysisReport(
        meta=UnifiedReportMeta(
            query=query,
            mode=mode,
            documents_analyzed=docs_analyzed,
            retrieval_strategy=strategy,
            model_used=model,
        ),
        verdict=SentimentVerdict(
            overall_sentiment="Mixed",
            confidence=0.3,
            net_sentiment_score=0,
            one_line_summary="Analysis completed with limited structured data.",
        ),
        executive_summary=raw_text[:500] if raw_text else "Analysis could not be fully structured. Please try again.",
        positive_signals=SignalBlock(
            headline="Some positive signals detected.",
            percentage=33.0,
        ),
        negative_signals=SignalBlock(
            headline="Some concerns identified.",
            percentage=34.0,
        ),
        sentiment_distribution=SentimentDistribution(
            positive_pct=33.0,
            negative_pct=34.0,
            neutral_pct=33.0,
            dominant_emotions=["mixed"],
        ),
        trend_indicators=TrendIndicators(
            sentiment_trajectory="Insufficient Data",
        ),
    )
    return report.model_dump()


# ============================================================================
# LEGACY — kept for backwards compatibility during migration window
# ============================================================================

class Theme(BaseModel):
    name: str
    sentiment: str
    frequency: str
    description: str


class Quote(BaseModel):
    text: str
    context: str
    sentiment: str
    score: int = 0


class SentimentBreakdown(BaseModel):
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    dominant_emotions: List[str]
    sarcasm_detected: bool


class ReportMeta(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    documents_analyzed: int = 0
    retrieval_strategy: str = "hybrid"
    model_used: str = "gemini-2.0-flash"


class SentimentReport(BaseModel):
    """LEGACY — use UnifiedAnalysisReport for all new code."""
    query: str
    overall_sentiment: str
    confidence: float
    summary: str
    key_themes: List[Theme] = Field(default_factory=list)
    notable_quotes: List[Quote] = Field(default_factory=list)
    sentiment_breakdown: SentimentBreakdown
    meta: ReportMeta = Field(default_factory=ReportMeta)


class QuickSearchResult(BaseModel):
    """LEGACY — use UnifiedAnalysisReport for all new code."""
    query: str
    answer: str
    sources: list = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
