# 🤖 Reddit Sentiment RAG: Autoresearch Optimization Agent

This module implements an autonomous prompt optimization engine based on Andrej Karpathy's `autoresearch` concept. It automatically optimizes target LLM system prompts using a **git-backed ratchet loop**.

---

## 🏗️ Architecture

```
                       [Start]
                          │
                          ▼
             [Run Baseline Evaluation] (Gemini 3.1 Flash Lite)
                          │
                          ▼
            ┌──► [Query Proposer Agent] (Gemma 4 31B)
            │             │  (Suggests prompt updates)
            │             ▼
            │     [Overwrite Prompt]
            │             │
            │             ▼
            │    [Run Evaluation Suite] (Gemini 3.1 Flash Lite)
            │             │  (Computes Score: QA + Latency + Retries)
            │             ▼
            │      [Is Score Better?]
            │         /         \
            │       Yes          No
            │       /             \
            │      ▼               ▼
            │ [git commit]    [git checkout --] (revert)
            └──────┴───────────────┘
```

---

## 🛠️ Components

1. **[`scratch/eval_pipeline.py`](file:///c:/Users/nithi/OneDrive/Desktop/Antigravity/Reddit-Sentiment-RAG/scratch/eval_pipeline.py)**
   - Runs a set of fixed queries against populated database collections (`GooglePixel` and `playstation`).
   - Forces RAG-only routing (mocking query intelligence routing) to prevent triggering browser scraper Playwright setups, keeping evaluations fast (approx. 30-40s total).
   - Uses an **LLM Judge** (`gemini-3.1-flash-lite`) to grade report relevance, structural constraints (count of quotes/insights), and actionable quality on a scale of 0 to 30.
   - Computes a final metric:
     $$\text{Final Score} = \text{Average Judge Score} - (2 \times \text{Reflection Retries}) - (15 \times \text{Pipeline Failures}) - (0.05 \times \text{Latency Seconds})$$

2. **[`scratch/autoresearch_runner.py`](file:///c:/Users/nithi/OneDrive/Desktop/Antigravity/Reddit-Sentiment-RAG/scratch/autoresearch_runner.py)**
   - Executes the Karpathy optimization loop.
   - Instantiates **Gemma 4 31B** via Google AI Studio (`gemma-4-31b-it`) to serve as the prompt editor.
   - Rewrites [`prompts/synthesizer.txt`](file:///c:/Users/nithi/OneDrive/Desktop/Antigravity/Reddit-Sentiment-RAG/prompts/synthesizer.txt) to maximize schema compliance and quality score.
   - Interacts with local Git index to commit successful increments and rollback failures.

---

## 🚀 How to Run

### 1. Requirements
- Ensure your environment variables are configured in [`.env`](file:///c:/Users/nithi/OneDrive/Desktop/Antigravity/Reddit-Sentiment-RAG/.env):
  ```env
  GOOGLE_API_KEY=your_google_ai_studio_key
  ```
- Make sure your local Git working tree is **clean** before running the optimizer (otherwise, local uncommitted modifications to other files might be overwritten during Git rollbacks).

### 2. Verify the Evaluation Script
Run the evaluation benchmark once manually to ensure it correctly connects to local databases and runs RAG:
```bash
python scratch/eval_pipeline.py
```

### 3. Launch the Optimization Loop
Run the Autoresearch loop:
```bash
python scratch/autoresearch_runner.py
```

It will execute 5 iterations. You will see live Git diffs and score updates in the terminal console.
If it finds a better prompt configuration, it commits it as a new Git commit.
If it regresses, it automatically reverts the changes.
