# Feature Spec: Subreddit Validation and Confirmation

This feature improves the scraping workflow by validating the existence of a subreddit before starting the scrape process and requiring user confirmation.

## Requirements

1. **Subreddit Existence Check**:
    - Before scraping, the system must verify if the subreddit exists.
    - If the subreddit does not exist (e.g., returns 404 or is private), inform the user and prompt for a correct name.

2. **User Confirmation**:
    - If the subreddit is valid, display key information about it (title, subscriber count, description).
    - Request explicit confirmation from the user before proceeding to scrape.

3. **Backend API**:
    - Add an endpoint `POST /api/validate_subreddit` that takes a subreddit name and returns its metadata or an error.

4. **Frontend Integration**:
    - Intercept the "Start Scraping" action.
    - Show a validation state (loading).
    - Show an error message if invalid.
    - Show a confirmation modal/panel if valid.
    - Only trigger `POST /api/scrape` after confirmation.

## Technical Implementation

### Backend (`scraper/reddit_scraper.py`)
- Implement `validate_subreddit(subreddit_name: str) -> dict`.
- Use `https://www.reddit.com/r/{subreddit}/about.json`.
- Handle 404 (not found), 403 (private), and other network errors.

### Backend (`app.py`)
- Add `POST /api/validate_subreddit`.
- Use a Pydantic model for the request.

### Frontend (`static/index.html` / JS)
- Update the scraping logic to include the validation step.
- Implement a confirmation UI.

## Verification Plan

1. **Invalid Subreddit**:
    - Input: `ManchesterUnit`
    - Expected: Error message "Subreddit 'ManchesterUnit' does not exist."
2. **Valid Subreddit**:
    - Input: `ManchesterUnited`
    - Expected: Display "r/ManchesterUnited: [Title] - [Subscribers] subscribers. Proceed?"
3. **Scrape after Confirmation**:
    - Action: Click "Confirm"
    - Expected: Scraping starts as usual.
