"""
Pydantic Schemas for Structured Sentiment Output
-------------------------------------------------
These schemas enforce strict structure on the LLM's output,
turning qualitative Reddit arguments into quantitative, trackable data.
"""

from datetime import datetime
from pydantic import BaseModel, Field


class Theme(BaseModel):
    """A major theme or topic identified in the discussion."""
    name: str = Field(description="Short name for the theme (e.g., 'Manager Criticism', 'Transfer Window')")
    sentiment: str = Field(description="Sentiment for this theme: Positive, Negative, Mixed, or Neutral")
    frequency: str = Field(description="How common this theme is: Common, Rare, or Universal")
    description: str = Field(description="1-2 sentence description of this theme and what people are saying")


class Quote(BaseModel):
    """A notable direct quote from a Reddit comment."""
    text: str = Field(description="The exact quote from the comment")
    context: str = Field(description="Brief context about what the quote is responding to")
    sentiment: str = Field(description="Sentiment of this quote: Positive, Negative, Mixed, or Neutral")
    score: int = Field(default=0, description="Reddit score (upvotes) of the comment")


class SentimentBreakdown(BaseModel):
    """Detailed numerical breakdown of sentiment distribution."""
    positive_pct: float = Field(ge=0, le=100, description="Percentage of positive sentiment (0-100)")
    negative_pct: float = Field(ge=0, le=100, description="Percentage of negative sentiment (0-100)")
    neutral_pct: float = Field(ge=0, le=100, description="Percentage of neutral sentiment (0-100)")
    dominant_emotions: list[str] = Field(
        description="Top 3-5 emotions detected (e.g., 'anger', 'frustration', 'hope', 'sarcasm')"
    )
    sarcasm_detected: bool = Field(description="Whether sarcasm or irony was detected in the comments")


class ReportMeta(BaseModel):
    """Metadata about how the report was generated."""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    documents_analyzed: int = Field(description="Number of documents retrieved and analyzed")
    retrieval_strategy: str = Field(default="hybrid", description="Which retrieval strategy was used")
    model_used: str = Field(default="gemini-2.0-flash", description="LLM model used for analysis")


class SentimentReport(BaseModel):
    """Complete sentiment analysis report."""
    query: str = Field(description="The original user query")
    overall_sentiment: str = Field(description="Overall sentiment: Positive, Negative, Mixed, or Neutral")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    summary: str = Field(description="2-3 sentence executive summary of the sentiment analysis")
    key_themes: list[Theme] = Field(description="Major themes identified in the discussion (3-6 themes)")
    notable_quotes: list[Quote] = Field(description="3-5 notable direct quotes that represent the sentiment")
    sentiment_breakdown: SentimentBreakdown = Field(description="Numerical breakdown of sentiment distribution")
    meta: ReportMeta = Field(
        default_factory=lambda: ReportMeta(documents_analyzed=0),
        description="Metadata about report generation"
    )


class QuickSearchResult(BaseModel):
    """Simplified result for quick RAG search (Phase 1 chain)."""
    query: str
    answer: str
    sources: list[dict] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
