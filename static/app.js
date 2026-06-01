// Global State
let currentMode = 'deep';
let currentStrategy = 'agentic';
let currentStep = 1;
let scrapePollInterval = null;
let embedPollInterval = null;
let currentCollection = 'reddit_sentiment';
let currentReportId = null;

// Chart.js Instances
let sentimentChartInstance = null;
let emotionChartInstance = null;
let compareSentimentChart = null;
let compareEmotionChart = null;

let collectionsList = [];

// Init on Document Load
document.addEventListener('DOMContentLoaded', () => {
    // Initial fetch of background state
    fetchCollections();
    fetchHistory();
    fetchFileCount();

    // Event listeners
    document.getElementById('searchInput').addEventListener('keydown', e => { if (e.key === 'Enter') runAnalysis(); });
    
    // Header badge synchronization
    const subInput = document.getElementById('scrapeSubreddit');
    const syncBadge = () => {
        const v = subInput.value.trim();
        document.getElementById('headerBadge').textContent = v ? `r/${v}` : 'r/community';
    };
    if (subInput) {
        subInput.addEventListener('input', syncBadge);
        syncBadge();
    }


});

// Helper: Debounce utility for searches
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ── Tab Management ───────────────────────────────────────────
function switchTab(tabId) {
    // Set tabs active states
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.toggle('active', tab.id === `${tabId}-tab`);
    });

    // Set tab contents active
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabId}-content`);
    });

    // Sub-routes trigger data loading
    if (tabId === 'compare') {
        populateComparisonSubreddits();

    } else if (tabId === 'rag') {
        if (currentStep === 3) {
            fetchStats();
            fetchWordCloud();
        }
    }
}

// Strategy and Mode Buttons
function setStrategy(strategy) {
    currentStrategy = strategy;
    document.querySelectorAll('[data-strategy]').forEach(b => b.classList.toggle('active', b.dataset.strategy === strategy));
}

function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('[data-mode]').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
}

// ── Stepper Management ───────────────────────────────────────
function goToStep(step) {
    if (step < 1 || step > 3) return;
    currentStep = step;

    // Update Step panels
    document.querySelectorAll('.panel').forEach((p, i) => {
        p.classList.toggle('active', (i + 1) === step);
    });

    // Update stepper header circles
    document.querySelectorAll('.step-item').forEach((item, i) => {
        const idx = i + 1;
        item.classList.toggle('active', idx === step);
        item.classList.toggle('completed', idx < step);
    });

    const wrapper = document.getElementById('analysisResultsWrapper');
    if (wrapper) {
        wrapper.style.display = (step === 3) ? 'block' : 'none';
    }

    if (step === 3) {
        fetchStats();
        fetchWordCloud();
    }
    if (step === 2) fetchFileCount();
}

// ── Background Data Fetchers ─────────────────────────────────
async function fetchWordCloud() {
    try {
        const res = await fetch(`/api/wordcloud?collection_name=${encodeURIComponent(currentCollection)}`);
        const data = await res.json();
        const container = document.getElementById('wordcloudContainer');
        const section = document.getElementById('wordcloudSection');
        
        if (data.words && data.words.length > 0) {
            section.style.display = 'block';
            const maxCount = Math.max(...data.words.map(w => w.value));
            const minCount = Math.min(...data.words.map(w => w.value));
            
            const colors = [
                'var(--accent-orange)', 'var(--accent-steel)', 'var(--accent-green)', 
                'var(--accent-amber)', 'var(--text-primary)', 'var(--text-secondary)'
            ];
            
            container.innerHTML = data.words.map(w => {
                const size = maxCount === minCount ? 1 : 0.8 + ((w.value - minCount) / (maxCount - minCount)) * 1.5;
                const color = colors[Math.floor(Math.random() * colors.length)];
                return `<span style="font-size: ${size}rem; color: ${color}; font-weight: ${Math.random() > 0.5 ? 700 : 400}; padding: 2px 6px; display: inline-block; transition: transform 0.2s; cursor: pointer" onmouseover="this.style.transform='scale(1.1)'" onmouseout="this.style.transform='scale(1)'" onclick="rerunQuery('${w.text}')">${w.text}</span>`;
            }).join('');
        } else {
            section.style.display = 'none';
        }
    } catch (e) {
        console.error('Wordcloud error:', e);
    }
}

async function fetchFileCount() {
    try {
        const res = await fetch(`/api/data/count?subreddit=${encodeURIComponent(currentCollection)}`);
        const data = await res.json();
        const el = document.getElementById('jsonFileCount');
        if (el) el.textContent = data.count || '0';
    } catch (e) { console.error('Count error:', e); }
}

async function fetchCollections() {
    try {
        const res = await fetch('/api/collections');
        const data = await res.json();
        collectionsList = data.collections || [];

        // Populate dropdown on Step 1
        const select = document.getElementById('existingDbSelect');
        const section = document.getElementById('existingDbSection');
        
        if (collectionsList.length > 0) {
            if (section) section.style.display = 'block';
            if (select) {
                select.innerHTML = '<option value="">-- Select a Subreddit --</option>' + 
                    collectionsList.map(c => `<option value="${c.name}">r/${c.name} (${c.docs} docs)</option>`).join('');
            }
        } else {
            if (section) section.style.display = 'none';
        }
    } catch (e) { console.error('Collections error:', e); }
}

function useExistingDb() {
    const select = document.getElementById('existingDbSelect');
    if (!select || !select.value) return;
    currentCollection = select.value;
    document.getElementById('headerBadge').textContent = `r/${currentCollection}`;
    fetchStats();
    goToStep(3);
}

// ── Reddit Scraper Controls ──────────────────────────────────
let pendingScrapeData = null;

async function startScraping() {
    const subreddit = document.getElementById('scrapeSubreddit').value.trim();
    if (!subreddit) return;

    const btn = document.getElementById('startScrapeBtn');
    const errorDiv = document.getElementById('errorMsg');
    const confDiv = document.getElementById('subredditConfirmation');

    btn.classList.add('loading');
    btn.disabled = true;
    errorDiv.classList.remove('visible');
    confDiv.style.display = 'none';

    try {
        const res = await fetch('/api/validate_subreddit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subreddit })
        });

        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.error || 'Validation failed');
        }

        const meta = data.metadata;
        document.getElementById('confSubName').textContent = `r/${meta.name}`;
        document.getElementById('confSubTitle').textContent = meta.title || '';
        document.getElementById('confSubCount').textContent = meta.subscribers?.toLocaleString() || '0';
        document.getElementById('confSubDesc').textContent = meta.description || 'No description available.';
        
        currentCollection = meta.name;
        
        pendingScrapeData = {
            subreddit: meta.name,
            post_limit: parseInt(document.getElementById('scrapeLimit').value),
            sort_by: document.getElementById('scrapeSort').value,
            time_filter: document.getElementById('scrapeTime').value
        };

        confDiv.style.display = 'block';
        confDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (e) {
        errorDiv.textContent = `❌ ${e.message}`;
        errorDiv.classList.add('visible');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

function cancelScrape() {
    document.getElementById('subredditConfirmation').style.display = 'none';
    pendingScrapeData = null;
}

async function confirmAndScrape() {
    if (!pendingScrapeData) return;

    const btn = document.getElementById('startScrapeBtn');
    const confDiv = document.getElementById('subredditConfirmation');
    const progress = document.getElementById('scrapeProgress');

    confDiv.style.display = 'none';
    btn.classList.add('loading');
    btn.disabled = true;
    progress.style.display = 'block';
    
    document.getElementById('scrapeBarFill').style.width = '0%';
    document.getElementById('scrapePercentText').textContent = '0%';
    document.getElementById('scrapeStatusText').textContent = 'Starting...';

    try {
        const res = await fetch('/api/scrape', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(pendingScrapeData)
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Scraping failed to start');
        }

        pollScrapeStatus();
    } catch (e) {
        const errorDiv = document.getElementById('errorMsg');
        errorDiv.textContent = `❌ ${e.message}`;
        errorDiv.classList.add('visible');
        btn.classList.remove('loading');
        btn.disabled = false;
        progress.style.display = 'none';
    }
}

function pollScrapeStatus() {
    if (scrapePollInterval) clearInterval(scrapePollInterval);

    scrapePollInterval = setInterval(async () => {
        try {
            const res = await fetch('/api/scrape/status');
            const data = await res.json();

            const bar = document.getElementById('scrapeBarFill');
            const percentText = document.getElementById('scrapePercentText');
            const statusText = document.getElementById('scrapeStatusText');
            const detailText = document.getElementById('scrapeDetailText');

            const pct = data.progress || 0;
            bar.style.width = pct + '%';
            percentText.textContent = Math.round(pct) + '%';
            statusText.textContent = data.running ? 'Scraping...' : (data.error ? 'Error' : 'Completed');
            detailText.textContent = data.message || 'Processing...';

            if (!data.running) {
                clearInterval(scrapePollInterval);
                document.getElementById('startScrapeBtn').classList.remove('loading');
                document.getElementById('startScrapeBtn').disabled = false;

                if (!data.error) {
                    setTimeout(() => {
                        goToStep(2);
                    }, 1000);
                }
            }
        } catch (e) {
            console.error('Poll error:', e);
        }
    }, 1000);
}

// ── Embedder Controls ────────────────────────────────────────
async function startEmbedding() {
    const btn = document.getElementById('startEmbedBtn');
    const progress = document.getElementById('embedProgress');
    const errorDiv = document.getElementById('errorMsg');

    btn.classList.add('loading');
    btn.disabled = true;
    errorDiv.classList.remove('visible');
    progress.style.display = 'block';

    try {
        const res = await fetch('/api/embed', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subreddit: currentCollection || pendingScrapeData?.subreddit })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Embedding failed to start');
        }

        pollEmbedStatus();
    } catch (e) {
        errorDiv.textContent = `❌ ${e.message}`;
        errorDiv.classList.add('visible');
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

function pollEmbedStatus() {
    if (embedPollInterval) clearInterval(embedPollInterval);

    embedPollInterval = setInterval(async () => {
        try {
            const res = await fetch('/api/embed/status');
            const data = await res.json();

            const bar = document.getElementById('embedBarFill');
            const percentText = document.getElementById('embedPercentText');
            const statusText = document.getElementById('embedStatusText');
            const detailText = document.getElementById('embedDetailText');

            const pct = data.progress || 0;
            bar.style.width = pct + '%';
            percentText.textContent = Math.round(pct) + '%';
            statusText.textContent = data.running ? 'Embedding...' : (data.error ? 'Error' : 'Completed');
            detailText.textContent = data.message || 'Processing...';

            if (!data.running) {
                clearInterval(embedPollInterval);
                document.getElementById('startEmbedBtn').classList.remove('loading');
                document.getElementById('startEmbedBtn').disabled = false;

                if (!data.error) {
                    setTimeout(() => {
                        fetchStats();
                        fetchCollections();
                        goToStep(3);
                    }, 1000);
                }
            }
        } catch (e) {
            console.error('Poll error:', e);
        }
    }, 1000);
}

// ── Stats Fetcher ────────────────────────────────────────────
async function fetchStats() {
    try {
        const res = await fetch(`/api/stats?collection_name=${encodeURIComponent(currentCollection)}`);
        const data = await res.json();
        
        document.getElementById('statDocs').textContent = data.total_documents?.toLocaleString() || '0';
        document.getElementById('statDate').textContent = data.date_range ? `${data.date_range.earliest?.slice(5) || '?'} → ${data.date_range.latest?.slice(5) || '?'}` : '—';
        document.getElementById('statScore').textContent = data.avg_score || '0';
        document.getElementById('statFlairs').textContent = data.flair_distribution ? Object.keys(data.flair_distribution).length : '0';

        if (!data.api_key_configured) {
            const warning = document.getElementById('apiWarning');
            const keyName = document.getElementById('missingKeyName');
            const keyLink = document.getElementById('keyLink');

            if (data.provider === 'Google AI Studio') {
                keyName.textContent = 'GOOGLE_API_KEY';
                keyLink.textContent = 'Google AI Studio';
                keyLink.href = 'https://aistudio.google.com/app/apikey';
            } else {
                keyName.textContent = 'OPENROUTER_API_KEY';
                keyLink.textContent = 'OpenRouter';
                keyLink.href = 'https://openrouter.ai/keys';
            }
            warning.classList.add('visible');
        } else {
            const warning = document.getElementById('apiWarning');
            if (warning) warning.classList.remove('visible');
        }
    } catch (e) { console.error('Stats error:', e); }
}

// ── History Retrieval ────────────────────────────────────────
async function fetchHistory() {
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        const container = document.getElementById('historyContainer');
        if (!data.history?.length) return;
        
        container.innerHTML = data.history.slice(0, 10).map(h => {
            const sentClass = (h.sentiment || '').toLowerCase();
            const time = new Date(h.timestamp).toLocaleString();
            const clickHandler = h.report_id ? `loadReport('${h.report_id}')` : `rerunQuery('${h.query.replace(/'/g, "\\'")}')`;
            let dateRangeHtml = '';
            if (h.data_freshness && h.data_freshness.earliest_post && h.data_freshness.latest_post) {
                dateRangeHtml = `<div style="font-size: 0.8rem; color: #888; margin-top: 4px;">📅 ${h.data_freshness.earliest_post.slice(5)} → ${h.data_freshness.latest_post.slice(5)}</div>`;
            }
            return `<div class="history-item" onclick="${clickHandler}">
                <span class="history-query">${h.query}</span>
                <span class="history-time">${time}</span>
                ${h.sentiment !== 'N/A' ? `<span class="history-sentiment sentiment-badge ${sentClass}">${h.sentiment}</span>` : ''}
                ${dateRangeHtml}
            </div>`;
        }).join('');
    } catch (e) { console.error('History error:', e); }
}

function rerunQuery(query) {
    document.getElementById('searchInput').value = query;
    goToStep(3);
    runAnalysis();
}

async function loadReport(reportId) {
    const btn = document.getElementById('searchBtn');
    const skeleton = document.getElementById('loadingSkeleton');
    const errorDiv = document.getElementById('errorMsg');

    btn.classList.add('loading');
    btn.disabled = true;
    skeleton.style.display = 'block';
    errorDiv.classList.remove('visible');
    document.getElementById('unifiedResults').classList.remove('visible');
    document.getElementById('quickResult').style.display = 'none';

    try {
        const res = await fetch(`/api/report/${reportId}`);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to load report');
        }

        const data = await res.json();
        
        if (data.report && data.report.meta && data.report.meta.query) {
            document.getElementById('searchInput').value = data.report.meta.query;
        }
        
        goToStep(3);
        renderUnifiedReport(data.mode, data.report);
    } catch (e) {
        errorDiv.textContent = `❌ ${e.message}`;
        errorDiv.classList.add('visible');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
        skeleton.style.display = 'none';
    }
}

// ── RAG Analysis Job Runner ──────────────────────────────────
async function runAnalysis() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) return;

    const btn = document.getElementById('searchBtn');
    const skeleton = document.getElementById('loadingSkeleton');
    const errorDiv = document.getElementById('errorMsg');
    const statusText = document.getElementById('analysisStatusText');

    btn.classList.add('loading');
    btn.disabled = true;
    skeleton.style.display = 'block';
    if (statusText) statusText.textContent = 'Starting analysis...';
    
    // Clear and reset console
    const consoleDiv = document.getElementById('analysisConsole');
    if (consoleDiv) {
        consoleDiv.innerHTML = '<div style="color:var(--text-muted)">[SYSTEM] Initializing console stream...</div>';
    }
    
    errorDiv.classList.remove('visible');
    document.getElementById('unifiedResults').classList.remove('visible');
    document.getElementById('quickResult').style.display = 'none';

    try {
        const res = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, mode: currentMode, collection: currentCollection, strategy: currentStrategy })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Analysis failed to start');
        }

        const data = await res.json();
        const jobId = data.job_id;
        
        const eventSource = new EventSource(`/api/analyze/stream/${jobId}`);
        
        eventSource.onmessage = async (event) => {
            const eData = JSON.parse(event.data);
            if (statusText) statusText.textContent = eData.message || 'Analyzing...';
            
            // Feed console box
            const consoleDiv = document.getElementById('analysisConsole');
            if (consoleDiv && eData.message) {
                const timestamp = new Date(eData.timestamp ? eData.timestamp * 1000 : Date.now()).toLocaleTimeString();
                const logLine = document.createElement('div');
                
                let badgeColor = 'var(--text-secondary)';
                const msgUpper = (eData.message || '').toUpperCase();
                const isRetry = msgUpper.includes('RATE LIMIT') || msgUpper.includes('QUOTA') || msgUpper.includes('60 SECONDS') || msgUpper.includes('RETRY');
                const isFallback = msgUpper.includes('FALLBACK') || msgUpper.includes('FALLING BACK') || msgUpper.includes('FALLS BACK');
                
                if (eData.status === 'completed') {
                    badgeColor = 'var(--accent-green)';
                } else if (eData.status === 'failed') {
                    badgeColor = 'var(--accent-red)';
                } else if (eData.status === 'running') {
                    if (isRetry) {
                        badgeColor = 'var(--accent-amber)';
                    } else if (isFallback) {
                        badgeColor = 'var(--accent-red)';
                    } else if (eData.message.includes('Agent 1')) {
                        badgeColor = 'var(--accent-steel)';
                    } else if (eData.message.includes('Agent 2') || eData.message.includes('Agent 3') || eData.message.includes('Agent 4') || eData.message.includes('Reflection')) {
                        badgeColor = 'var(--accent-orange)';
                    }
                }
                
                logLine.innerHTML = `<span style="color:var(--text-muted)">[${timestamp}]</span> <span style="color:${badgeColor}; font-weight: 600;">[${eData.status.toUpperCase()}]</span> ${eData.message}`;
                
                if (isRetry) {
                    logLine.style.borderLeft = '3px solid var(--accent-amber)';
                    logLine.style.paddingLeft = '6px';
                    logLine.style.background = 'rgba(193, 125, 46, 0.08)';
                    logLine.style.borderRadius = '2px';
                } else if (isFallback) {
                    logLine.style.borderLeft = '3px solid var(--accent-red)';
                    logLine.style.paddingLeft = '6px';
                    logLine.style.background = 'rgba(192, 57, 43, 0.08)';
                    logLine.style.borderRadius = '2px';
                }
                
                consoleDiv.appendChild(logLine);
                consoleDiv.scrollTop = consoleDiv.scrollHeight;
            }
            
            if (eData.status === 'completed' || eData.status === 'failed') {
                eventSource.close();
                if (eData.status === 'failed') {
                    errorDiv.textContent = `❌ ${eData.error || 'Analysis failed'}`;
                    errorDiv.classList.add('visible');
                    btn.classList.remove('loading');
                    btn.disabled = false;
                    skeleton.style.display = 'none';
                } else {
                    // Retry fetching status up to 5 times to handle race conditions
                    // where the result may not be stored yet when the SSE event fires.
                    const fetchResult = async (retries = 5, delayMs = 500) => {
                        for (let i = 0; i < retries; i++) {
                            try {
                                const statusRes = await fetch(`/api/analyze/status/${jobId}`);
                                if (!statusRes.ok) throw new Error(`Status ${statusRes.status}`);
                                const statusData = await statusRes.json();
                                if (statusData.result && statusData.result.report) {
                                    return statusData.result;
                                }
                                // Result not ready yet — wait and retry
                            } catch (err) {
                                if (i === retries - 1) throw err;
                            }
                            await new Promise(r => setTimeout(r, delayMs));
                        }
                        throw new Error('Report result unavailable after multiple retries');
                    };

                    try {
                        const result = await fetchResult();
                        renderUnifiedReport(result.mode, result.report);
                        fetchHistory();
                    } catch (e) {
                        console.error('Error rendering report:', e);
                        errorDiv.textContent = `❌ Failed to load final report: ${e.message}`;
                        errorDiv.classList.add('visible');
                    } finally {
                        btn.classList.remove('loading');
                        btn.disabled = false;
                        skeleton.style.display = 'none';
                    }
                }
            }
        };
        
        eventSource.onerror = (err) => {
            eventSource.close();
            errorDiv.textContent = `❌ Lost connection to analysis stream.`;
            errorDiv.classList.add('visible');
            btn.classList.remove('loading');
            btn.disabled = false;
            skeleton.style.display = 'none';
        };
    } catch (e) {
        errorDiv.textContent = `❌ ${e.message}`;
        errorDiv.classList.add('visible');
        btn.classList.remove('loading');
        btn.disabled = false;
        skeleton.style.display = 'none';
    }
}

// ── Unified Analysis Report Renderer ─────────────────────────
function renderUnifiedReport(mode, r) {
    if (!r) return;

    document.getElementById('quickResult').style.display = 'none';
    document.getElementById('unifiedResults').classList.remove('visible');

    if (mode === 'quick') {
        const sentClass = (r.sentiment || 'neutral').toLowerCase();
        const badge = document.getElementById('quickSentimentBadge');
        badge.textContent = r.sentiment || '—';
        badge.className = `sentiment-badge ${sentClass}`;

        document.getElementById('quickConfidenceText').textContent =
            `Confidence: ${((r.confidence || 0) * 100).toFixed(0)}%`;

        typewriter(document.getElementById('quickSummaryText'), r.summary || '');

        const meta = r.meta || {};
        const freshness = meta.data_freshness || {};
        
        document.getElementById('quickStrategyBadge').textContent = `🔀 ${meta.retrieval_strategy || 'N/A'}`;
        
        let modelDisplay = meta.model_used || 'N/A';
        if (meta.fallback_used) {
            modelDisplay += ` <span class="sentiment-badge negative" style="font-size: 0.65rem; padding: 2px 6px; margin: 0 0 0 4px; vertical-align: middle;">⚠️ Fallback Active</span>`;
        }
        if (meta.retries_occurred > 0) {
            modelDisplay += ` <span class="sentiment-badge mixed" style="font-size: 0.65rem; padding: 2px 6px; margin: 0 0 0 4px; vertical-align: middle;">⏳ ${meta.retries_occurred}x Retried</span>`;
        }

        document.getElementById('quickMetaInfo').innerHTML =
            `📢 r/${meta.collection || 'N/A'} &nbsp;|&nbsp;
             📄 ${meta.documents_analyzed || 0} docs &nbsp;|&nbsp;
             🔀 ${meta.retrieval_strategy || 'N/A'} &nbsp;|&nbsp;
             🤖 ${modelDisplay} &nbsp;|&nbsp;
             📅 ${freshness.earliest_post || '?'} → ${freshness.latest_post || '?'} &nbsp;|&nbsp;
             🕐 ${meta.timestamp ? new Date(meta.timestamp).toLocaleString() : 'N/A'}`;

        document.getElementById('quickResult').style.display = 'block';
        return;
    }

    const verdict = r.verdict || {};
    const dist = r.sentiment_distribution || {};
    const posS = r.positive_signals || {};
    const negS = r.negative_signals || {};

    // ① Verdict Header
    const sentClass = (verdict.overall_sentiment || 'neutral').toLowerCase();
    const badge = document.getElementById('sentimentBadge');
    badge.textContent = verdict.overall_sentiment || '—';
    badge.className = `sentiment-badge ${sentClass}`;

    const net = verdict.net_sentiment_score || 0;
    const netBadge = document.getElementById('netScoreBadge');
    netBadge.textContent = net > 0 ? 'Positive' : (net < 0 ? 'Negative' : 'Neutral');
    netBadge.className = `net-score-badge ${net > 0 ? 'pos' : net < 0 ? 'neg' : 'zero'}`;

    document.getElementById('modeBadgeLabel').textContent = '🧠 DEEP';
    
    const reportMeta = r.meta || {};
    document.getElementById('strategyBadgeLabel').textContent = `🔀 ${reportMeta.retrieval_strategy || 'N/A'}`;
    
    document.getElementById('oneLinerText').textContent = verdict.one_line_summary || '';
    document.getElementById('confidenceText').textContent = `Confidence: ${((verdict.confidence || 0) * 100).toFixed(0)}%`;
    typewriter(document.getElementById('summaryText'), r.executive_summary || '');

    // ② Signals
    document.getElementById('posPct').textContent = Math.round(posS.percentage || 0) + '%';
    document.getElementById('posHeadline').textContent = posS.headline || '';
    document.getElementById('posThemes').innerHTML = renderThemeRows(posS.top_themes || []);
    document.getElementById('posQuotes').innerHTML = renderSignalQuotes(posS.praise_quotes || [], 'positive');

    document.getElementById('negPct').textContent = Math.round(negS.percentage || 0) + '%';
    document.getElementById('negHeadline').textContent = negS.headline || '';
    document.getElementById('negThemes').innerHTML = renderThemeRows(negS.top_themes || []);
    document.getElementById('negQuotes').innerHTML = renderSignalQuotes(negS.criticism_quotes || [], 'negative');

    // ③ Sentiment Distribution & Emotion Map Render (Chart.js!)
    document.getElementById('pctPositive').textContent = Math.round(dist.positive_pct || 0);
    document.getElementById('pctNegative').textContent = Math.round(dist.negative_pct || 0);
    document.getElementById('pctNeutral').textContent = Math.round(dist.neutral_pct || 0);
    document.getElementById('emotionsContainer').innerHTML = (dist.dominant_emotions || []).map(e => `<span class="emotion-tag">${e}</span>`).join('');

    setTimeout(() => {
        document.getElementById('barPositive').style.width = (dist.positive_pct || 0) + '%';
        document.getElementById('barNegative').style.width = (dist.negative_pct || 0) + '%';
        document.getElementById('barNeutral').style.width = (dist.neutral_pct || 0) + '%';
        const cs = (dist.controversy_score || 0) * 100;
        document.getElementById('controversyFill').style.width = cs + '%';
    }, 100);

    // Initialize Interactive Charts
    initRAGCharts(dist);

    // Dynamic Consensus Badge
    const cScore = dist.controversy_score || 0;
    const consensusBadge = document.getElementById('consensusBadge');
    if (consensusBadge) {
        let text = '';
        let bg = '';
        let color = '';

        if (cScore < 0.40) {
            if (verdict.overall_sentiment === 'Positive') {
                text = '✨ United Approval'; bg = 'rgba(16, 185, 129, 0.15)'; color = 'var(--accent-green)';
            } else if (verdict.overall_sentiment === 'Negative') {
                text = '⚠️ United Disapproval'; bg = 'rgba(239, 68, 68, 0.15)'; color = 'var(--accent-red)';
            } else {
                text = '🤝 Broad Consensus'; bg = 'rgba(245, 158, 11, 0.15)'; color = 'var(--accent-amber)';
            }
        } else if (cScore >= 0.70) {
            text = '⚔️ Polarized / Heated'; bg = 'rgba(244, 63, 94, 0.15)'; color = '#f43f5e';
        } else {
            text = verdict.overall_sentiment === 'Mixed' ? '⚖️ Diverse Views' : '📈 Mild Division';
            bg = 'rgba(136, 136, 170, 0.15)'; color = 'var(--text-secondary)';
        }

        consensusBadge.textContent = text;
        consensusBadge.style.background = bg;
        consensusBadge.style.color = color;
        consensusBadge.style.border = `1px solid ${color.replace(')', ', 0.3)')}`;
        consensusBadge.style.display = 'inline-block';
    }

    // Controversy drivers
    const driversContainer = document.getElementById('controversyDriversContainer');
    const driversList = document.getElementById('controversyDriversList');
    if (driversContainer && driversList) {
        const drivers = dist.controversy_drivers || [];
        if (drivers.length > 0) {
            driversList.innerHTML = drivers.map(d => 
                `<div style="font-size:0.78rem; color:var(--text-secondary); line-height:1.4;">
                    <span style="color:var(--accent-orange); margin-right:6px;">•</span>${d}
                 </div>`
            ).join('');
            driversContainer.style.display = 'block';
        } else {
            driversContainer.style.display = 'none';
        }
    }

    // ④ Key Entities
    const entColors = { Positive:'var(--accent-green)', Negative:'var(--accent-red)', Mixed:'var(--accent-amber)', Neutral:'var(--text-secondary)' };
    document.getElementById('entitiesContainer').innerHTML = (r.key_entities || []).map(e =>
        `<span class="entity-pill" title="${e.mention_count} mentions">
            <span class="entity-dot" style="background:${entColors[e.net_sentiment]||'#8b5cf6'}"></span>
            ${e.name}
            <span class="entity-type-badge">${e.type}</span>
            <span class="entity-count">×${e.mention_count}</span>
        </span>`).join('');

    // ⑤ Competitive & Trends
    const cs = r.competitive_signals || {};
    document.getElementById('competitiveContainer').innerHTML = `
        <div class="comp-row"><span class="comp-key">Competitor mentions</span><span class="comp-val">${cs.mentions_competitors ? '✅ Yes' : '➖ None'}</span></div>
        ${(cs.competitors_mentioned||[]).length ? `<div class="comp-row"><span class="comp-key">Competitors</span><span class="comp-val">${cs.competitors_mentioned.join(', ')}</span></div>` : ''}
        <div class="comp-row"><span class="comp-key">Comparison sentiment</span><span class="comp-val">${cs.comparison_sentiment || 'N/A'}</span></div>`;

    const ti = r.trend_indicators || {};
    const traj = ti.sentiment_trajectory || 'Insufficient Data';
    const trajClass = ['Improving','Declining','Stable'].includes(traj) ? `trend-${traj}` : 'trend-default';
    document.getElementById('trendContainer').innerHTML = `
        <div style="margin-bottom:10px"><span class="trend-badge ${trajClass}">${traj}</span></div>
        ${(ti.urgent_concerns||[]).map(c=>`<div class="concern-item">${c}</div>`).join('')}
        ${(ti.emerging_positives||[]).map(p=>`<div class="positive-item">${p}</div>`).join('')}`;

    // ⑥ Actionable insights
    const ai = r.actionable_insights || {};
    const teams = [
        { key:'for_product_team', icon:'🛠️', label:'Product Team' },
        { key:'for_marketing_team', icon:'📣', label:'Marketing Team' },
        { key:'for_support_team', icon:'🎧', label:'Support Team' },
    ];
    document.getElementById('insightsContainer').innerHTML = teams.map(t => {
        const items = (ai[t.key] || []).map(i => `<div class="insight-item">${i}</div>`).join('');
        return `<div class="insight-team open" onclick="this.classList.toggle('open')">
            <div class="insight-team-header">
                <span class="insight-team-title">${t.icon} ${t.label}</span>
                <span class="insight-chevron">▼</span>
            </div>
            <div class="insight-team-body">${items || '<div class="insight-item">No recommendations.</div>'}</div>
        </div>`;
    }).join('');

    // ⑦ Report Metadata & Downloads
    const meta = r.meta || {};
    const freshness = meta.data_freshness || {};
    currentReportId = meta.report_id || null;
    if (currentReportId) {
        document.getElementById('downloadReportBtn').style.display = 'inline-block';
    } else {
        document.getElementById('downloadReportBtn').style.display = 'none';
    }
    
    let modelDisplay = meta.model_used || 'N/A';
    if (meta.fallback_used) {
        modelDisplay += ` <span class="sentiment-badge negative" style="font-size: 0.65rem; padding: 2px 6px; margin: 0 0 0 4px; vertical-align: middle;">⚠️ Fallback Active</span>`;
    }
    if (meta.retries_occurred > 0) {
        modelDisplay += ` <span class="sentiment-badge mixed" style="font-size: 0.65rem; padding: 2px 6px; margin: 0 0 0 4px; vertical-align: middle;">⏳ ${meta.retries_occurred}x Retried</span>`;
    }

    document.getElementById('metaInfo').innerHTML =
        `🆔 ${meta.report_id ? meta.report_id.slice(0,8)+'...' : 'N/A'} &nbsp;|&nbsp;
         📢 r/${meta.collection || 'N/A'} &nbsp;|&nbsp;
         📄 ${meta.documents_analyzed||0} docs &nbsp;|&nbsp;
         🤖 ${modelDisplay} &nbsp;|&nbsp;
         🔀 ${meta.retrieval_strategy||'N/A'} &nbsp;|&nbsp;
         📅 ${freshness.earliest_post||'?'} → ${freshness.latest_post||'?'} &nbsp;|&nbsp;
         🕐 ${meta.timestamp ? new Date(meta.timestamp).toLocaleString() : 'N/A'}`;

    document.getElementById('unifiedResults').classList.add('visible');
}

// ── Chart.js Lifecycle: Analysis Renders ─────────────────────
function initRAGCharts(dist) {
    // 1. Sentiment Distribution Doughnut
    const ctxSentiment = document.getElementById('sentimentChart');
    if (ctxSentiment) {
        if (sentimentChartInstance) sentimentChartInstance.destroy();
        
        sentimentChartInstance = new Chart(ctxSentiment, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    data: [dist.positive_pct || 0, dist.negative_pct || 0, dist.neutral_pct || 0],
                    backgroundColor: ['#2E8B57', '#C0392B', '#9A8F87'],
                    borderColor: 'rgba(22, 20, 18, 0.6)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#EDE9E4', font: { family: 'Inter', size: 11 } }
                    }
                },
                cutout: '65%'
            }
        });
    }

    // 2. Emotion Map Horizontal Bar
    const ctxEmotion = document.getElementById('emotionChart');
    if (ctxEmotion) {
        if (emotionChartInstance) emotionChartInstance.destroy();

        const emap = dist.emotion_map || {};
        const labels = Object.keys(emap);
        const values = Object.values(emap);

        const emotionColors = { 
            anger:'#ef4444', frustration:'#f97316', hope:'#10b981',
            satisfaction:'#22d3ee', disappointment:'#8b5cf6', excitement:'#f59e0b',
            sarcasm:'#ec4899', resignation:'#6b7280' 
        };
        const bgColors = labels.map(l => emotionColors[l.toLowerCase()] || '#8b5cf6');

        emotionChartInstance = new Chart(ctxEmotion, {
            type: 'bar',
            data: {
                labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
                datasets: [{
                    data: values,
                    backgroundColor: bgColors,
                    borderWidth: 0,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(232, 210, 195, 0.05)' },
                        ticks: { color: '#9A8F87', font: { family: 'JetBrains Mono', size: 10 } },
                        max: 100
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#EDE9E4', font: { family: 'Inter', size: 11 } }
                    }
                }
            }
        });
    }
}

// Subrender Helpers
function renderThemeRows(themes) {
    return themes.map(t => {
        const q = t.representative_quote || {};
        return `<div class="theme-row">
            <div class="theme-row-name">${t.theme}<span class="theme-freq">${t.frequency}</span></div>
            <div class="theme-row-desc">${t.description}</div>
            ${q.text ? `<div class="theme-quote">"${q.text}"<span class="theme-quote-score">Upvotes: ${q.score||0}</span></div>` : ''}
        </div>`;
    }).join('');
}

function renderSignalQuotes(quotes, type) {
    return quotes.map(q => `<div class="signal-quote">
        <div class="signal-quote-text">"${q.text}"</div>
        <div class="signal-quote-meta">
            <span style="color:${type==='positive'?'var(--accent-green)':'var(--accent-red)'}">⬆️ ${q.score||0}</span>
            <span>${q.context||''}</span>
            ${q.source_post ? `<span style="opacity:0.6">${q.source_post}</span>` : ''}
        </div>
    </div>`).join('');
}

function downloadCurrentReport() {
    if (!currentReportId) return;
    window.open(`/api/report/${currentReportId}/download?format=html`, '_blank');
}

function typewriter(el, text) {
    el.textContent = '';
    let i = 0;
    const interval = setInterval(() => {
        if (i < text.length) { el.textContent += text[i]; i++; }
        else clearInterval(interval);
    }, 8);
}

// ── Subreddit Comparison Dashboard ───────────────────────────
function populateComparisonSubreddits() {
    const subA = document.getElementById('compareSubA');
    const subB = document.getElementById('compareSubB');
    if (!subA || !subB) return;

    const options = collectionsList.map(c => `<option value="${c.name}">r/${c.name} (${c.docs} docs)</option>`).join('');
    subA.innerHTML = '<option value="">-- Select Subreddit A --</option>' + options;
    subB.innerHTML = '<option value="">-- Select Subreddit B --</option>' + options;
}

async function runComparison() {
    const subA = document.getElementById('compareSubA').value;
    const subB = document.getElementById('compareSubB').value;
    
    if (!subA || !subB) {
        alert("Please select two subreddits to compare.");
        return;
    }
    if (subA === subB) {
        alert("Please select two different subreddits.");
        return;
    }

    const loader = document.getElementById('compareLoader');
    const results = document.getElementById('compareResults');
    const errorDiv = document.getElementById('compareError');

    loader.style.display = 'block';
    results.style.display = 'none';
    errorDiv.style.display = 'none';

    try {
        // Fetch raw corpus statistics
        const [resA, resB, resHist] = await Promise.all([
            fetch(`/api/stats?collection_name=${encodeURIComponent(subA)}`),
            fetch(`/api/stats?collection_name=${encodeURIComponent(subB)}`),
            fetch('/api/history')
        ]);

        const statsA = await resA.json();
        const statsB = await resB.json();
        const historyData = await resHist.json();

        // Find latest Deep Analysis reports from history for Subreddit A and Subreddit B
        const history = historyData.history || [];
        const reportAInfo = history.find(h => h.report_id && h.mode === 'deep' && h.data_freshness && h.query.toLowerCase().includes(subA.toLowerCase()));
        const reportBInfo = history.find(h => h.report_id && h.mode === 'deep' && h.data_freshness && h.query.toLowerCase().includes(subB.toLowerCase()));

        let reportA = null;
        let reportB = null;

        if (reportAInfo) {
            const rA = await fetch(`/api/report/${reportAInfo.report_id}`);
            if (rA.ok) reportA = (await rA.json()).report;
        }
        if (reportBInfo) {
            const rB = await fetch(`/api/report/${reportBInfo.report_id}`);
            if (rB.ok) reportB = (await rB.json()).report;
        }

        // Render Side-by-Side Metadata
        renderCompareColumns(subA, subB, statsA, statsB, reportA, reportB);

        // Render Comparative Charts (Grouped Bars + Radar)
        renderCompareCharts(subA, subB, reportA, reportB);

        loader.style.display = 'none';
        results.style.display = 'block';
    } catch (e) {
        loader.style.display = 'none';
        errorDiv.textContent = `Error comparing subreddits: ${e.message}`;
        errorDiv.style.display = 'block';
    }
}

function renderCompareColumns(subA, subB, statsA, statsB, reportA, reportB) {
    // Fill Sub A Info Card
    document.getElementById('compTitleA').textContent = `r/${subA}`;
    document.getElementById('compDocsA').textContent = statsA.total_documents?.toLocaleString() || '0';
    document.getElementById('compDateA').textContent = statsA.date_range ? `${statsA.date_range.earliest?.slice(5) || '?'} → ${statsA.date_range.latest?.slice(5) || '?'}` : 'N/A';
    document.getElementById('compScoreA').textContent = statsA.avg_score || '0';
    
    // Fill Sub B Info Card
    document.getElementById('compTitleB').textContent = `r/${subB}`;
    document.getElementById('compDocsB').textContent = statsB.total_documents?.toLocaleString() || '0';
    document.getElementById('compDateB').textContent = statsB.date_range ? `${statsB.date_range.earliest?.slice(5) || '?'} → ${statsB.date_range.latest?.slice(5) || '?'}` : 'N/A';
    document.getElementById('compScoreB').textContent = statsB.avg_score || '0';

    // Set Verdicts from Deep RAG Report
    const rSentA = reportA?.verdict?.overall_sentiment || 'No RAG analysis';
    const rSentB = reportB?.verdict?.overall_sentiment || 'No RAG analysis';
    const sentBadgeA = document.getElementById('compSentA');
    const sentBadgeB = document.getElementById('compSentB');
    
    sentBadgeA.textContent = rSentA;
    sentBadgeA.className = `sentiment-badge ${rSentA.toLowerCase()}`;
    sentBadgeB.textContent = rSentB;
    sentBadgeB.className = `sentiment-badge ${rSentB.toLowerCase()}`;

    document.getElementById('compOneLinerA').textContent = reportA?.verdict?.one_line_summary || 'Run a deep report on this community to synthesize qualitative insights.';
    document.getElementById('compOneLinerB').textContent = reportB?.verdict?.one_line_summary || 'Run a deep report on this community to synthesize qualitative insights.';

    // Top Positive & Negative Drivers
    document.getElementById('compDriverPosA').innerHTML = reportA?.positive_signals?.headline ? `<strong>✨ Praise Driver:</strong> ${reportA.positive_signals.headline}` : 'No RAG positive signals loaded.';
    document.getElementById('compDriverPosB').innerHTML = reportB?.positive_signals?.headline ? `<strong>✨ Praise Driver:</strong> ${reportB.positive_signals.headline}` : 'No RAG positive signals loaded.';

    document.getElementById('compDriverNegA').innerHTML = reportA?.negative_signals?.headline ? `<strong>⚠️ Concern Driver:</strong> ${reportA.negative_signals.headline}` : 'No RAG negative signals loaded.';
    document.getElementById('compDriverNegB').innerHTML = reportB?.negative_signals?.headline ? `<strong>⚠️ Concern Driver:</strong> ${reportB.negative_signals.headline}` : 'No RAG negative signals loaded.';
}

function renderCompareCharts(subA, subB, reportA, reportB) {
    const ctxCompareSentiment = document.getElementById('compareSentimentChart');
    const ctxCompareEmotion = document.getElementById('compareEmotionChart');

    if (compareSentimentChart) compareSentimentChart.destroy();
    if (compareEmotionChart) compareEmotionChart.destroy();

    // 1. Sentiment Distribution Comparison Chart
    const distA = reportA?.sentiment_distribution || { positive_pct: 0, negative_pct: 0, neutral_pct: 0 };
    const distB = reportB?.sentiment_distribution || { positive_pct: 0, negative_pct: 0, neutral_pct: 0 };

    if (ctxCompareSentiment) {
        compareSentimentChart = new Chart(ctxCompareSentiment, {
            type: 'bar',
            data: {
                labels: ['Positive %', 'Negative %', 'Neutral %'],
                datasets: [
                    {
                        label: `r/${subA}`,
                        data: [distA.positive_pct, distA.negative_pct, distA.neutral_pct],
                        backgroundColor: 'rgba(232, 73, 15, 0.75)',
                        borderColor: '#E8490F',
                        borderWidth: 1,
                        borderRadius: 3
                    },
                    {
                        label: `r/${subB}`,
                        data: [distB.positive_pct, distB.negative_pct, distB.neutral_pct],
                        backgroundColor: 'rgba(61, 126, 191, 0.75)',
                        borderColor: '#3D7EBF',
                        borderWidth: 1,
                        borderRadius: 3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#EDE9E4' } }
                },
                scales: {
                    x: { ticks: { color: '#EDE9E4' }, grid: { display: false } },
                    y: { ticks: { color: '#9A8F87' }, grid: { color: 'rgba(255,255,255,0.05)' }, max: 100 }
                }
            }
        });
    }

    // 2. Emotion Map Comparison Radar Chart
    const emapA = distA.emotion_map || {};
    const emapB = distB.emotion_map || {};
    const allEmotions = ['anger', 'frustration', 'hope', 'satisfaction', 'disappointment', 'excitement', 'sarcasm', 'resignation'];
    
    const valA = allEmotions.map(e => emapA[e] || 0);
    const valB = allEmotions.map(e => emapB[e] || 0);

    if (ctxCompareEmotion) {
        compareEmotionChart = new Chart(ctxCompareEmotion, {
            type: 'radar',
            data: {
                labels: allEmotions.map(e => e.charAt(0).toUpperCase() + e.slice(1)),
                datasets: [
                    {
                        label: `r/${subA}`,
                        data: valA,
                        backgroundColor: 'rgba(232, 73, 15, 0.15)',
                        borderColor: '#E8490F',
                        pointBackgroundColor: '#E8490F',
                        borderWidth: 2
                    },
                    {
                        label: `r/${subB}`,
                        data: valB,
                        backgroundColor: 'rgba(61, 126, 191, 0.15)',
                        borderColor: '#3D7EBF',
                        pointBackgroundColor: '#3D7EBF',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#EDE9E4' } }
                },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
                        grid: { color: 'rgba(255, 255, 255, 0.08)' },
                        pointLabels: { color: '#EDE9E4', font: { size: 10 } },
                        ticks: { color: '#9A8F87', backdropColor: 'transparent', font: { size: 9 } },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }
}


