# Project Implementation TODO

Tracking the implementation status of features defined in `.agents/specs/`.

## Core Infrastructure
- [x] Set up SDD Workflow (MASTER_SPEC.md)
- [x] Initialize `TODO.md` tracking <!-- mapped to MASTER_SPEC.md -->

## Feature Specs
- [x] Subreddit Validation and Confirmation <!-- mapped to features/subreddit_validation.md -->
- [x] Enriched Analysis Output — Quick & Deep Mode <!-- mapped to features/enriched_analysis_output.md (FEAT-004) -->
- [x] Quick Mode Conversational Summary + Router Fix <!-- mapped to features/quick_mode_and_router_fix.md (FEAT-005 + BUGFIX-001) -->
- [ ] Latency Optimization <!-- mapped to features/latency_optimization.md -->

## Implementation Progress
| Feature | Spec File | Status |
| :--- | :--- | :--- |
| SDD Setup | `MASTER_SPEC.md` | Completed |
| Scraping Fix | `features/fix_scraping_logic.md` | Completed |
| Subreddit Validation | `features/subreddit_validation.md` | Completed |
| Enriched Analysis Output | `features/enriched_analysis_output.md` | ✅ Implemented |
| Quick Mode Redesign | `features/quick_mode_and_router_fix.md` | ✅ Implemented |
| Router Fix (BUGFIX-001) | `features/quick_mode_and_router_fix.md` | ✅ Fixed |
| Router Visibility & Word Cloud | `features/router_visibility_wordcloud.md` | ✅ Implemented |
| Latency Optimization | `features/latency_optimization.md` | ✅ Implemented |


## FEAT-004 Phase Checklist
### Phase 1: Schema Foundation
- [x] Add new Pydantic models to `agents/schemas.py`
- [x] Verify JSON serialization round-trip

### Phase 2: Quick Mode Upgrade
- [x] Rewrite `QUICK_ANALYSIS_SYSTEM_PROMPT` in `rag/generator.py`
- [x] Update `query_rag()` return type from `str` → `dict`
- [x] Add `_parse_and_enrich()` helper
- [x] Increase `n_results` default from 5 → 8

### Phase 3: Deep Mode Upgrade
- [x] Update `EXTRACTOR_PROMPT` (domain-agnostic, positive/negative split)
- [x] Update `SENTIMENT_PROMPT` (emotion map, controversy, trajectory)
- [x] Rewrite `SYNTHESIZER_PROMPT` → full `UnifiedAnalysisReport` JSON
- [x] Update `synthesize()` post-processing (derived fields, data_freshness)

### Phase 4: API Contract
- [x] Update `/api/analyze` response envelope in `app.py`
- [x] Update history entry fields

### Phase 5: Frontend Rendering
- [x] Add PositiveSignals card (green)
- [x] Add NegativeSignals card (red)
- [x] Add EmotionMap visualisation
- [x] Add KeyEntities section
- [x] Add ActionableInsights accordion
- [x] Add TrendIndicators section

## Latency Optimization Checklist
### Phase 1: Critical Fixes
- [x] Fix BM25 on-the-fly build in `rag/retriever.py`
- [x] Cache graph compilation in `agents/pipeline.py`

### Phase 2: Parallelization & I/O Optimization
- [x] Parallelize multi-query retrieval in `rag/retriever.py`
- [x] Stream data in `rag/embedder.py`

### Phase 3: Advanced Optimization
- [x] Combine Router and Multi-Query LLM calls in `rag/retriever.py`
