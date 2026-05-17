# Router Visibility and Word Cloud

## Objective
Enhance RAG transparency by clearly displaying the agentic router's strategy choice and provide user inspiration by showing a word cloud of the embedded data before a query is entered.

## Requirements

### 1. Agentic Router Visibility
- The currently selected retrieval strategy (Semantic vs Hybrid) must be displayed prominently in the analysis results.
- **Frontend changes**: Update `index.html` to display the `retrieval_strategy` dynamically. A badge or an explicit label should be added next to the confidence score or in a prominent summary section, instead of being hidden at the very bottom in the meta information.

### 2. Embedded Data Word Cloud
- Display a word cloud of key terms from the selected dataset (Chroma collection) in the UI before the user types their query.
- **Backend changes**:
  - Add an endpoint `GET /api/wordcloud?collection_name=<name>` in `app.py`.
  - The endpoint should fetch documents from ChromaDB, tokenize them, filter out common English stopwords, and calculate word frequencies.
  - Return the top 30-50 words with their frequencies.
- **Frontend changes**:
  - Add a container in `index.html` within the Analyze step (Step 3) to render the word cloud.
  - Add a Javascript function to fetch the word cloud data when a collection is selected or the Analyze step is activated.
  - Render the words using varying font sizes based on frequency (e.g., using inline styles and mapping frequencies to a `min..max` rem scale).

## Implementation Details
1. **Stopwords filtering**: Use a simple hardcoded list of stopwords in Python to avoid adding large NLP dependencies.
2. **Word Cloud UI**: Make it aesthetically pleasing, matching the Reddit Sentiment Intelligence dark mode theme. Warm charcoal colors, accents (orange, green, red, amber, steel) can be randomly assigned to words.
