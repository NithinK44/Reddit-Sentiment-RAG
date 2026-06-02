# FEATURE SPEC: Enriched Analysis Output — Quick & Deep Mode
**Spec ID**: `FEAT-004`
**Status**: `PENDING`
**Author**: Antigravity AI
**Created**: 2026-05-15
**Mapped To**: `TODO.md` → "Enriched Analysis Output"

---

## 1. Business Context & Motivation

This system is intended to serve as a **Product Intelligence Platform** — enabling companies to monitor what real users are saying about their products, services, or brand on Reddit. Today, the analysis output is shallow:

- **Quick Mode** returns a single unstructured markdown string with no machine-readable structure.
- **Deep Mode** returns a structured JSON report, but it is narrow: only a summary, a few themes, a few quotes, and a percentage breakdown.

Neither mode explicitly separates **positive feedback** from **negative feedback**, which is the #1 need for a product team reviewing community sentiment.

**Goal**: Reshape both modes to return a **rich, uniform, structured intelligence report** that a company analyst can act on immediately — without needing to read through raw Reddit threads.

---

## 2. Target Users

| User | Use Case |
|---|---|
| Product Manager | Track satisfaction with a product release |
| Brand Manager | Monitor brand perception over time |
| Customer Support Lead | Identify recurring pain points and complaints |
| Marketing Team | Find praise and organic advocacy quotes to amplify |
| Executive Stakeholder | Get a snapshot confidence score and sentiment |

---

## 3. Core Design Principles

1. **Positive/Negative Symmetry**: Every output MUST include an equal-depth analysis of both positive and negative signals. No asymmetry.
2. **Actionability**: Every section must answer a concrete question a business analyst would ask.
3. **Quote-Anchored**: Claims must be grounded in real quotes from the data. No unsourced assertions.
4. **Uniform Schema**: Both Quick and Deep modes must conform to the same canonical `UnifiedAnalysisReport` schema so the UI can render them identically.
5. **Deterministic Structure**: Every run must return the exact same top-level keys. Missing fields should be `null` or `[]`, never absent.
6. **Domain-Agnostic Prompts**: All LLM prompts must NOT hard-code "football" or "ManchesterUnited". They must use `{topic}` placeholders so the same pipeline works for any product/brand/subreddit.

---

## 4. Canonical Output Schema: `UnifiedAnalysisReport`

This is the single contract between the backend and the frontend. Both Quick and Deep modes MUST produce this structure.

```json
{
  "meta": {
    "report_id": "<uuid4>",
    "query": "<original user query>",
    "mode": "quick | deep",
    "timestamp": "<ISO 8601>",
    "documents_analyzed": 0,
    "retrieval_strategy": "semantic | hybrid",
    "model_used": "<model name>",
    "data_freshness": {
      "earliest_post": "<date or null>",
      "latest_post": "<date or null>"
    }
  },

  "verdict": {
    "overall_sentiment": "Positive | Negative | Mixed | Neutral",
    "confidence": 0.0,
    "net_sentiment_score": 0,
    "one_line_summary": "<≤ 20 word punchy headline>"
  },

  "executive_summary": "<3-4 sentence business-readable summary covering both positives and negatives>",

  "positive_signals": {
    "headline": "<1 sentence — what people love>",
    "percentage": 0,
    "top_themes": [
      {
        "theme": "<theme name>",
        "frequency": "Universal | Common | Rare",
        "evidence_count": 0,
        "description": "<what exactly users are praising, 1-2 sentences>",
        "representative_quote": {
          "text": "<exact quote>",
          "score": 0,
          "source_post": "<post title or null>"
        }
      }
    ],
    "praise_quotes": [
      {
        "text": "<exact quote>",
        "score": 0,
        "context": "<what this is responding to>",
        "source_post": "<post title or null>"
      }
    ]
  },

  "negative_signals": {
    "headline": "<1 sentence — what people dislike>",
    "percentage": 0,
    "top_themes": [
      {
        "theme": "<theme name>",
        "frequency": "Universal | Common | Rare",
        "evidence_count": 0,
        "description": "<what exactly users are criticizing, 1-2 sentences>",
        "representative_quote": {
          "text": "<exact quote>",
          "score": 0,
          "source_post": "<post title or null>"
        }
      }
    ],
    "criticism_quotes": [
      {
        "text": "<exact quote>",
        "score": 0,
        "context": "<what this is responding to>",
        "source_post": "<post title or null>"
      }
    ]
  },

  "sentiment_distribution": {
    "positive_pct": 0,
    "negative_pct": 0,
    "neutral_pct": 0,
    "dominant_emotions": [],
    "emotion_map": {
      "anger": 0,
      "frustration": 0,
      "hope": 0,
      "satisfaction": 0,
      "disappointment": 0,
      "excitement": 0,
      "sarcasm": 0,
      "resignation": 0
    },
    "sarcasm_detected": false,
    "controversy_score": 0
  },

  "key_entities": [
    {
      "name": "<entity name — product, person, feature, event>",
      "type": "Product | Person | Feature | Event | Other",
      "mention_count": 0,
      "net_sentiment": "Positive | Negative | Mixed | Neutral"
    }
  ],

  "competitive_signals": {
    "mentions_competitors": false,
    "competitors_mentioned": [],
    "comparison_sentiment": "Favourable | Unfavourable | Neutral | N/A"
  },

  "trend_indicators": {
    "sentiment_trajectory": "Improving | Declining | Stable | Insufficient Data",
    "urgent_concerns": [],
    "emerging_positives": []
  },

  "actionable_insights": {
    "for_product_team": [],
    "for_marketing_team": [],
    "for_support_team": []
  }
}
```

---

## 5. Schema: Pydantic Models (`agents/schemas.py`)

### 5.1 New / Modified Models

All existing models become **deprecated** and replaced by the unified schema below. Implement as additive classes — do NOT delete `SentimentReport` until UI migration is complete.

```python
# New Pydantic classes to ADD to agents/schemas.py

class ReportMeta(BaseModel):
    report_id: str
    query: str
    mode: str  # "quick" | "deep"
    timestamp: str
    documents_analyzed: int
    retrieval_strategy: str
    model_used: str
    data_freshness: dict  # {"earliest_post": str|None, "latest_post": str|None}

class SentimentVerdict(BaseModel):
    overall_sentiment: str  # Positive|Negative|Mixed|Neutral
    confidence: float       # 0.0–1.0
    net_sentiment_score: int  # computed: positive_pct - negative_pct
    one_line_summary: str

class SignalTheme(BaseModel):
    theme: str
    frequency: str          # Universal|Common|Rare
    evidence_count: int
    description: str
    representative_quote: dict  # {text, score, source_post}

class SignalQuote(BaseModel):
    text: str
    score: int
    context: str
    source_post: Optional[str]

class SignalBlock(BaseModel):
    headline: str
    percentage: float
    top_themes: List[SignalTheme]
    praise_quotes: List[SignalQuote] = []     # used by positive block
    criticism_quotes: List[SignalQuote] = []  # used by negative block

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
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    dominant_emotions: List[str]
    emotion_map: EmotionMap
    sarcasm_detected: bool
    controversy_score: float  # 0.0–1.0 (high = heavily polarized community)

class KeyEntity(BaseModel):
    name: str
    type: str  # Product|Person|Feature|Event|Other
    mention_count: int
    net_sentiment: str

class CompetitiveSignals(BaseModel):
    mentions_competitors: bool
    competitors_mentioned: List[str]
    comparison_sentiment: str  # Favourable|Unfavourable|Neutral|N/A

class TrendIndicators(BaseModel):
    sentiment_trajectory: str  # Improving|Declining|Stable|Insufficient Data
    urgent_concerns: List[str]
    emerging_positives: List[str]

class ActionableInsights(BaseModel):
    for_product_team: List[str]
    for_marketing_team: List[str]
    for_support_team: List[str]

class UnifiedAnalysisReport(BaseModel):
    meta: ReportMeta
    verdict: SentimentVerdict
    executive_summary: str
    positive_signals: SignalBlock
    negative_signals: SignalBlock
    sentiment_distribution: SentimentDistribution
    key_entities: List[KeyEntity]
    competitive_signals: CompetitiveSignals
    trend_indicators: TrendIndicators
    actionable_insights: ActionableInsights
```

---

## 6. Changes: `rag/generator.py` (Quick Mode)

### 6.1 Current Behaviour
Returns a raw markdown string. No structure. No positive/negative split.

### 6.2 New Behaviour
`query_rag()` must return a `dict` conforming to `UnifiedAnalysisReport`.

### 6.3 New System Prompt for Quick Mode

The prompt must be **domain-agnostic** and instruct the LLM to output valid JSON matching the schema.

```
SYSTEM:
You are a Product Intelligence Analyst. You analyze user-generated content from online communities 
to help companies understand what real users think about their topics of interest.

You receive retrieved posts and comments. You MUST analyze BOTH positive AND negative signals 
with equal depth and rigor.

CRITICAL RULES:
1. ONLY use the provided context. Never fabricate quotes or data.
2. Every claim must be anchored to actual retrieved content.
3. Separate positive feedback from negative feedback explicitly.
4. Weight comments by their score — higher score = stronger community signal.
5. Detect sarcasm. A highly-upvoted sarcastic comment is a strong NEGATIVE signal.
6. Output ONLY valid JSON matching the schema provided. No markdown. No extra text.

SCHEMA: {schema_hint}

CONTEXT:
{context}
