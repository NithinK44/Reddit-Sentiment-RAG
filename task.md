# Task Tracker — Reddit Sentiment RAG Reorganization

## Phase 1: Create New Folder Structure
- [x] Create `scraper/` directory
- [x] Create `rag/` directory
- [x] Create `agents/` directory
- [x] Create `data/` directory
- [x] Create `docs/` directory

## Phase 2: Extract & Move Files
- [x] Extract scraper from `rough.ipynb` → `scraper/reddit_scraper.py`
- [x] Move `embed_to_chroma.py` → `rag/embedder.py` (update imports)
- [x] Move `retriever.py` → `rag/retriever.py` (update imports)
- [x] Move `generator.py` → `rag/generator.py` (update imports)
- [x] Move `agents.py` → `agents/pipeline.py` (update imports)
- [x] Move `schemas.py` → `agents/schemas.py` (update imports)
- [x] Move `implementation.md` → `docs/implementation.md`
- [x] Move `understanding.md` → `docs/understanding.md`
- [x] Create `__init__.py` files for packages
- [x] Migrate existing `chroma_db/` data to `data/chroma_db/`

## Phase 3: Update Core Config
- [x] Update `config.py` paths (DATA_DIR, CHROMA_PERSIST_DIR)
- [x] Update `.gitignore` for new structure

## Phase 4: Add Scraping API Endpoints
- [x] Add `POST /api/scrape` endpoint
- [x] Add `GET /api/scrape/status` endpoint
- [x] Add `POST /api/embed` endpoint
- [x] Update `app.py` imports for new module paths

## Phase 5: Rebuild Dashboard UI
- [x] Add multi-step workflow stepper (Scrape → Embed → Analyze)
- [x] Build Step 1: Scrape configuration panel
- [x] Build Step 2: Embed panel
- [x] Step 3: Existing analyze panel (update)
- [x] Add "Skip to Analysis" shortcut

## Phase 6: Cleanup
- [x] Delete `scratch/` directory (6 files)
- [x] Delete `test_endpoint.py`
- [x] Delete `rough.ipynb`
- [x] Delete old root-level copies of moved files
- [x] Delete old `chroma_db/` after migration

## Phase 7: Verify
- [x] Server starts without import errors
- [x] Dashboard loads and displays correctly
- [x] Scraping UI functional
- [x] Existing analysis flow still works
