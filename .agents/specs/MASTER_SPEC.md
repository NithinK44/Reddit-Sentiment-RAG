# MASTER SPECIFICATION

This document defines the Spec-Driven Development (SDD) protocol for the Reddit-Sentiment-RAG project. All development must adhere to these rules.

## SDD Protocol

### 1. Spec-First Development
- **Requirement**: Before writing any code, you must update or create a corresponding `feature.md` (or relevant spec file) in the `.agents/specs/` folder.
- **Goal**: Ensure every change is intentional and documented before implementation.

### 2. Mapping & Verification
- **Requirement**: Every code change must be mapped back to a specific line or section in the specification.
- **Verification**: The implementation must be verifiable against the spec requirements.

### 3. Token Efficiency (Delta-Only Coding)
- **Requirement**: Use 'Delta-Only' coding.
- **Action**: Do not rewrite entire files. Only provide the specific changes (diffs/snippets) needed to satisfy the spec.
- **Exception**: New files should be created in full.

### 4. Model Optimization
- **Requirement**: All logic and implementation must be optimized for **Gemini 3 Flash**.
- **Focus**: Efficiency, clarity, and leveraging Gemini's long-context and reasoning capabilities.

---

## Spec Directory Structure
- `.agents/specs/MASTER_SPEC.md`: This protocol.
- `.agents/specs/features/*.md`: Detailed specifications for individual features.
- `TODO.md`: Root-level tracking of implementation status.
