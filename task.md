# Task Tracker — Reddit Sentiment RAG Reorganization

## Phase 1: Create New Folder Structure
- [ ] Create `scraper/` directory
- [ ] Create `rag/` directory
- [ ] Create `agents/` directory
- [ ] Create `data/` directory
- [ ] Create `docs/` directory

## Phase 2: Extract & Move Files
- [ ] Extract scraper from `rough.ipynb` → `scraper/reddit_scraper.py`
- [ ] Move `embed_to_chroma.py` → `rag/embedder.py` (update imports)
- [ ] Move `retriever.py` → `rag/retriever.py` (update imports)
- [ ] Move `generator.py` → `rag/generator.py` (update imports)
- [ ] Move `agents.py` → `agents/pipeline.py` (update imports)
- [ ] Move `schemas.py` → `agents/schemas.py` (update imports)
- [ ] Move `implementation.md` → `docs/implementation.md`
- [ ] Move `understanding.md` → `docs/understanding.md`
- [ ] Create `__init__.py` files for packages
- [ ] Migrate existing `chroma_db/` data to `data/chroma_db/`

## Phase 3: Update Core Config
- [ ] Update `config.py` paths (DATA_DIR, CHROMA_PERSIST_DIR)
- [ ] Update `.gitignore` for new structure

## Phase 4: Add Scraping API Endpoints
- [ ] Add `POST /api/scrape` endpoint
- [ ] Add `GET /api/scrape/status` endpoint
- [ ] Add `POST /api/embed` endpoint
- [ ] Update `app.py` imports for new module paths

## Phase 5: Rebuild Dashboard UI
- [ ] Add multi-step workflow stepper (Scrape → Embed → Analyze)
- [ ] Build Step 1: Scrape configuration panel
- [ ] Build Step 2: Embed panel
- [ ] Step 3: Existing analyze panel (update)
- [ ] Add "Skip to Analysis" shortcut

## Phase 6: Cleanup
- [ ] Delete `scratch/` directory (6 files)
- [ ] Delete `test_endpoint.py`
- [ ] Delete `rough.ipynb`
- [ ] Delete old root-level copies of moved files
- [ ] Delete old `chroma_db/` after migration

## Phase 7: Verify
- [ ] Server starts without import errors
- [ ] Dashboard loads and displays correctly
- [ ] Scraping UI functional
- [ ] Existing analysis flow still works
