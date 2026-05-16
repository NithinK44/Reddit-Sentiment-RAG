# Feature Spec: Scraping Logic & UI Optimization

## 1. Requirements

### 1.1 "Best" Filter
- **UI**: Add a "Best" option to the "Sort By" dropdown in the Scrape panel.
- **Backend**: Map "best" to `best.json` (or `hot.json` if `best` is unavailable for subreddits).
- **Goal**: Allow users to use Reddit's "Best" sorting algorithm.

### 1.2 Performance Optimization
- **Rate Limiting**: Reduce `time.sleep(2)` to `time.sleep(1)` to double the scraping speed while remaining relatively safe.
- **Error Handling**: Add robust checks for non-JSON responses from Reddit API to prevent crashes.
- **Skip Logic**: Ensure that already scraped files are counted towards progress correctly and quickly.

### 1.3 Progress & Feedback
- **Feedback**: The `_scrape_state["message"]` should include the title of the current post being scraped.
- **UI Progress**: Ensure the progress bar starts moving immediately upon initiation.
- **Clarification**: Confirm that "Post Limit" in the UI dictates the total number of threads to be collected.

## 2. Technical Implementation (Delta-Only)

### 2.1 `static/index.html`
- Add `<option value="best">Best</option>` to `#scrapeSort`.
- Ensure `pollScrapeStatus` handles initial state transitions better.

### 2.2 `scraper/reddit_scraper.py`
- Update `scrape_subreddit` to handle `sort_by == 'best'`.
- Reduce sleep timer.
- Improve `_scrape_state["message"]` to provide granular feedback.

### 2.3 `app.py`
- No major changes needed, but ensure `scrape_subreddit` receives the new `best` parameter.

## 3. Verification Plan
- **Test**: Trigger a scrape with "Best" sort.
- **Test**: Verify the progress bar moves from 0% to 100%.
- **Test**: Verify that the "message" in the UI updates with post titles.
