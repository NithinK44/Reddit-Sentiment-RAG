# FEAT-005 + BUGFIX-001 — Quick Mode Redesign & Router Fix

## Status: IN PROGRESS

---

## Problem Statement

### BUGFIX-001: Semantic/Hybrid Router Import Failure
`rag/retriever.py` imports `MultiQueryRetriever` from `langchain_classic`, which does not
exist as a stable package. This causes an `ImportError` on startup or at analysis time,
breaking ALL retrieval (quick and deep).

**Root cause:** `from langchain_classic.retrievers.multi_query import MultiQueryRetriever`
**Fix:** Use `langchain_community.retrievers.multi_query.MultiQueryRetriever`
or more reliably, implement multi-query directly without the package dependency.

Additionally, the router's JSON output parser must be hardened to handle markdown-fenced
output from the LLM (```json ... ```).

### FEAT-005: Quick Mode Must Be Conversational Summary
Quick mode currently runs the SAME full `UnifiedAnalysisReport` JSON pipeline as Deep mode,
which is:
- Expensive (same LLM call cost as deep without the multi-agent reasoning)
- Confusing (renders all segments including emotions, entities, etc.)
- Not fit for purpose (should be a fast sanity check, not a full report)

**Required behaviour:**
- **Quick mode** backend: Single LLM call, returns a plain-text conversational summary.
  Output schema: `{ summary: str, sentiment: str, confidence: float, mode: "quick" }`
- **Quick mode** frontend: Renders ONE card — the verdict badge + a flowing paragraph.
  No segments, no themes, no emotion maps.
- **Deep mode**: Unchanged. Continues to return full `UnifiedAnalysisReport`.

---

## Acceptance Criteria

### BUGFIX-001
- [x] `retriever.py` no longer imports from `langchain_classic` or `langchain_community.retrievers`
- [x] `MultiQueryRetriever` replaced with inline `multi_query_retrieve()` function
- [x] Router JSON parser strips ```json fences before `json.loads`
- [x] Router has a robust fallback to SEMANTIC on any parsing failure

### FEAT-005
- [x] `query_rag()` in `rag/generator.py` returns `{ summary, sentiment, confidence, mode: "quick", meta }`
- [x] Quick mode prompt is conversational prose, NOT a JSON schema prompt
- [x] Frontend `renderUnifiedReport()` branches on `mode`:
  - `quick`: shows only verdict badge + conversational paragraph + meta footer
  - `deep`: unchanged full render
- [x] All deep mode segments are HIDDEN for quick mode responses

---

## Implementation Plan

### Step 1: Fix BUGFIX-001 (retriever.py)
- Replace `langchain_classic` import with `langchain_community`
- Harden router JSON parsing to strip markdown fences

### Step 2: Fix FEAT-005 backend (rag/generator.py)
- Rewrite `QUICK_ANALYSIS_SYSTEM_PROMPT` to be conversational prose, NOT JSON schema
- Change `query_rag()` return type to simple dict: `{summary, sentiment, confidence, meta}`

### Step 3: Update API response envelope (app.py)
- Quick mode response: `{mode: "quick", report: {summary, sentiment, confidence, meta}}`
- Deep mode response: unchanged `{mode: "deep", report: UnifiedAnalysisReport}`

### Step 4: Fix frontend (static/index.html)
- Branch `renderUnifiedReport(mode, r)` on mode
- Quick path: show verdict card + paragraph only, hide all segment divs
- Deep path: existing full render

---

## Files Modified
- `rag/retriever.py` — BUGFIX-001
- `rag/generator.py` — FEAT-005 backend
- `static/index.html` — FEAT-005 frontend
- `TODO.md` — tracking
