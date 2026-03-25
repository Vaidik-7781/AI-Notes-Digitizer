/**
 * AI Notes Digitizer — main.js
 * ════════════════════════════════════════════════════════════════════════════
 * Full frontend logic for the Flask-backed AI Notes Digitizer.
 *
 * Flask API endpoints consumed:
 *   POST /upload                → accept files → returns { file_ids: [] }
 *   POST /process               → start pipeline → returns { job_id }
 *   GET  /stream/<job_id>       → SSE progress stream
 *   GET  /preview/<job_id>      → formatted page data
 *   GET  /download/<job_id>     → DOCX file
 *   GET  /download-pdf/<job_id> → PDF file
 *   POST /reprocess/<job_id>    → re-run with new settings
 *   GET  /history               → session list
 *   POST /merge                 → merge job IDs → returns { merged_job_id }
 * ════════════════════════════════════════════════════════════════════════════
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════════════════
   CONFIG
   ═══════════════════════════════════════════════════════════════════════════ */
const API_BASE = 'http://127.0.0.1:5000/api'; // Point explicitly to the Flask backend on port 5000

/* ═══════════════════════════════════════════════════════════════════════════
   APPLICATION STATE
   ═══════════════════════════════════════════════════════════════════════════ */
const state = {
  /** @type {Array<{file:File, name:string, size:number, status:string, id:number}>} */
  files:          [],

  /** @type {string|null} */
  currentJobId:   null,

  /** @type {Array<object>} Formatted page objects from /preview */
  pages:          [],

  /** @type {number} 0-indexed */
  currentPage:    0,

  /** @type {boolean} */
  editMode:       false,

  /** @type {{theme:string, language:string, enhance_mode:string, add_summary:boolean, add_bullets:boolean}} */
  settings: {
    theme:        'academic',
    language:     'English',
    enhance_mode: 'auto',
    add_summary:  true,
    add_bullets:  true,
  },

  /** @type {Array<object>} */
  historyData:    [],

  /** @type {string[]} Session IDs selected for merge */
  mergeIds:       [],

  /** @type {EventSource|null} */
  sseSource:      null,

  /** @type {MediaStream|null} */
  cameraStream:   null,
};

/* ═══════════════════════════════════════════════════════════════════════════
   CONSTANTS
   ═══════════════════════════════════════════════════════════════════════════ */
const STEP_ORDER = ['enhance', 'ocr', 'formula', 'ai', 'docx'];
const STEP_ICONS = {
  enhance: 'image_search',
  ocr:     'scan',
  formula: 'functions',
  ai:      'psychology',
  docx:    'description',
};
const ALLOWED_TYPES = new Set([
  'image/jpeg', 'image/png', 'image/webp',
  'image/bmp',  'image/tiff', 'application/pdf',
]);
const ALLOWED_EXT_RE = /\.(jpg|jpeg|png|webp|bmp|tiff|tif|pdf)$/i;

/* ═══════════════════════════════════════════════════════════════════════════
   DOM HELPERS
   ═══════════════════════════════════════════════════════════════════════════ */
/** @param {string} id */
const $  = (id) => document.getElementById(id);
/** @param {string} sel */
const $$ = (sel) => document.querySelectorAll(sel);

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatBytes(bytes) {
  if (bytes < 1024)     return bytes + ' B';
  if (bytes < 1048576)  return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function formatDate(unixTs) {
  if (!unixTs) return '—';
  return new Date(unixTs * 1000).toLocaleString(undefined, {
    dateStyle: 'medium', timeStyle: 'short',
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   TOAST NOTIFICATIONS
   ═══════════════════════════════════════════════════════════════════════════ */
let _toastTimer = null;

/**
 * @param {string} msg
 * @param {'info'|'success'|'error'|'warn'} type
 * @param {number} [duration=3200]
 */
function showToast(msg, type = 'info', duration = 3200) {
  const el   = $('toast');
  const icon = $('toast-icon');
  const text = $('toast-msg');
  if (!el) return;

  const iconMap  = { info: 'info', success: 'check_circle', error: 'error', warn: 'warning' };
  const colorMap = { info: '#00f4fe', success: '#4ade80', error: '#ff716c', warn: '#ffc965' };

  icon.textContent = iconMap[type] || 'info';
  icon.style.color = colorMap[type] || colorMap.info;
  text.textContent = msg;

  el.className = `show toast-${type}`;
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { el.classList.remove('show'); }, duration);
}

/* ═══════════════════════════════════════════════════════════════════════════
   APP OVERLAY OPEN / CLOSE
   ═══════════════════════════════════════════════════════════════════════════ */
function openApp() {
  $('app-overlay')?.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeApp() {
  $('app-overlay')?.classList.remove('open');
  document.body.style.overflow = '';
}

/* ═══════════════════════════════════════════════════════════════════════════
   SETTINGS
   ═══════════════════════════════════════════════════════════════════════════ */
function toggleSettings() {
  const panel = $('settings-panel');
  if (!panel) return;
  panel.classList.toggle('open');
  const btn = $('settings-toggle');
  btn?.classList.toggle('active', panel.classList.contains('open'));
}

/**
 * @param {'theme'|'language'|'enhance_mode'|'add_summary'|'add_bullets'} key
 * @param {string|boolean} value
 */
function setSetting(key, value) {
  state.settings[key] = value;

  if (key === 'theme') {
    $$('.theme-btn').forEach(btn => {
      const active = btn.dataset.theme === value;
      btn.classList.toggle('active-theme', active);
    });
  }
  if (key === 'enhance_mode') {
    $$('.mode-btn').forEach(btn => {
      const active = btn.dataset.mode === value;
      btn.classList.toggle('active-mode', active);
    });
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   FILE QUEUE
   ═══════════════════════════════════════════════════════════════════════════ */
function handleFileSelect(e) {
  Array.from(e.target.files).forEach(addFileToQueue);
  e.target.value = '';
}

function handleDrop(e) {
  e.preventDefault();
  $('drop-zone')?.classList.remove('drag-over');
  Array.from(e.dataTransfer.files).forEach(addFileToQueue);
}

/** @param {File} file */
function addFileToQueue(file) {
  if (!ALLOWED_TYPES.has(file.type) && !ALLOWED_EXT_RE.test(file.name)) {
    showToast(`Unsupported file: ${file.name}`, 'error');
    return;
  }
  const entry = {
    file,
    name:   file.name,
    size:   file.size,
    type:   file.type,
    status: 'queued',
    id:     Date.now() + Math.random(),
  };
  state.files.push(entry);
  appendQueueItem(entry);
  $('queue-empty')?.classList.add('hidden');
  const btn = $('process-btn');
  if (btn) btn.disabled = false;
}

/** @param {{id:number, name:string, size:number, type:string, status:string}} entry */
function appendQueueItem(entry) {
  const container = $('file-queue');
  if (!container) return;

  const el = document.createElement('div');
  el.id = `qi-${entry.id}`;
  el.className = 'queue-item glass-panel';

  const isPdf = entry.type === 'application/pdf';
  el.innerHTML = `
    <div class="qi-file-icon" style="width:2rem;height:2rem;border-radius:.5rem;background:var(--surface-container-high);display:flex;align-items:center;justify-content:center;flex-shrink:0;">
      <span class="material-symbols-outlined font-fill" style="font-size:1.1rem;color:${isPdf ? 'var(--error)' : 'var(--primary)'};">
        ${isPdf ? 'picture_as_pdf' : 'image'}
      </span>
    </div>
    <div style="flex:1;min-width:0;">
      <p style="font-size:.75rem;font-weight:700;color:var(--on-surface);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${escHtml(entry.name)}</p>
      <p style="font-size:.65rem;color:var(--on-surface-variant);">${formatBytes(entry.size)}</p>
    </div>
    <div class="qi-status" id="qs-${entry.id}" style="flex-shrink:0;">${statusBadgeHTML('queued')}</div>
    <button onclick="removeFromQueue(${entry.id})" style="width:1.5rem;height:1.5rem;display:flex;align-items:center;justify-content:center;color:rgba(172,170,177,.4);cursor:pointer;background:none;border:none;flex-shrink:0;transition:color .2s;"
      onmouseover="this.style.color='var(--error)'" onmouseout="this.style.color='rgba(172,170,177,.4)'">
      <span class="material-symbols-outlined" style="font-size:.875rem;">close</span>
    </button>`;
  container.appendChild(el);
}

function updateQueueItemStatus(fileId, status) {
  const el = $(`qs-${fileId}`);
  if (el) el.innerHTML = statusBadgeHTML(status);
  const item = $(`qi-${fileId}`);
  if (item) {
    item.classList.toggle('done',  status === 'done');
    item.classList.toggle('error', status === 'error');
  }
}

/** @param {'queued'|'uploading'|'processing'|'done'|'error'} status */
function statusBadgeHTML(status) {
  const configs = {
    queued:     { icon: 'schedule',   color: 'rgba(172,170,177,.4)',         pulse: false },
    uploading:  { icon: 'upload',     color: 'var(--secondary)',             pulse: true  },
    processing: { icon: 'autorenew',  color: 'var(--primary)',               pulse: true  },
    done:       { icon: 'check_circle', color: '#4ade80',                    pulse: false },
    error:      { icon: 'error',      color: 'var(--error)',                 pulse: false },
  };
  const cfg = configs[status] || configs.queued;
  return `<span class="material-symbols-outlined font-fill" style="font-size:1.1rem;color:${cfg.color};${cfg.pulse ? 'animation:pulse-ring 1.5s infinite;' : ''}">${cfg.icon}</span>`;
}

/** @param {number} fileId */
function removeFromQueue(fileId) {
  state.files = state.files.filter(f => f.id !== fileId);
  $(`qi-${fileId}`)?.remove();
  if (state.files.length === 0) {
    $('queue-empty')?.classList.remove('hidden');
    const btn = $('process-btn');
    if (btn) btn.disabled = true;
  }
}

function clearQueue() {
  state.files = [];
  const container = $('file-queue');
  if (!container) return;
  container.innerHTML = `
    <div id="queue-empty" class="text-center" style="padding:1.5rem 0;color:rgba(172,170,177,.3);font-size:.875rem;text-align:center;">
      <span class="material-symbols-outlined" style="font-size:2rem;display:block;margin-bottom:.25rem;">inbox</span>No files queued
    </div>`;
  const btn = $('process-btn');
  if (btn) btn.disabled = true;
}

/* ═══════════════════════════════════════════════════════════════════════════
   UPLOAD  →  POST /upload
   ═══════════════════════════════════════════════════════════════════════════ */
async function uploadFiles() {
  const fd    = new FormData();
  let   count = 0;

  for (const f of state.files) {
    if (f.status === 'queued') {
      fd.append('files', f.file);
      f.status = 'uploading';
      updateQueueItemStatus(f.id, 'uploading');
      count++;
    }
  }
  if (count === 0) return null;

  try {
    const res  = await fetch(`${API_BASE}/upload`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Upload failed (${res.status})`);

    state.files.forEach(f => {
      if (f.status === 'uploading') {
        f.status = 'processing';
        updateQueueItemStatus(f.id, 'processing');
      }
    });
    return data;

  } catch (err) {
    state.files.forEach(f => {
      if (f.status === 'uploading') {
        f.status = 'error';
        updateQueueItemStatus(f.id, 'error');
      }
    });
    showToast(err.message, 'error');
    return null;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   PROCESS  →  POST /process
   ═══════════════════════════════════════════════════════════════════════════ */
async function startProcessing() {
  const btn = $('process-btn');
  if (btn) btn.disabled = true;

  showProgressArea(true);
  resetSteps();
  updateProgress(0, 'Uploading files…');
  hideResultPanel();

  const uploadData = await uploadFiles();
  if (!uploadData || !uploadData.files) {
    if (btn) btn.disabled = false;
    return;
  }

  const filePaths = uploadData.files.map(f => f.dest);

  try {
    const res  = await fetch(`${API_BASE}/process`, {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ 
        session_id: uploadData.session_id,
        file_paths: filePaths, 
        settings: state.settings 
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Process start failed (${res.status})`);

    state.currentJobId = data.job_id;
    connectSSE(data.job_id);

  } catch (err) {
    showToast(err.message, 'error');
    if (btn) btn.disabled = false;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   SSE STREAM  →  GET /stream/<job_id>
   ═══════════════════════════════════════════════════════════════════════════ */
function connectSSE(jobId) {
  if (state.sseSource) {
    state.sseSource.close();
    state.sseSource = null;
  }

  const url = `${API_BASE}/stream/${jobId}`;
  state.sseSource = new EventSource(url);

  state.sseSource.onmessage = (e) => {
    try { handleProgressMessage(JSON.parse(e.data)); }
    catch { /* ignore parse errors */ }
  };

  state.sseSource.onerror = () => {
    state.sseSource?.close();
    state.sseSource = null;
  };
}

/** @param {{status:string, stage?:string, progress?:number, error?:string}} msg */
function handleProgressMessage(msg) {
  const { status, stage, progress, error } = msg;
  const message = stage || error || '';

  let step = null;
  const stageLower = message.toLowerCase();
  if (stageLower.includes('enhance')) step = 'enhance';
  else if (stageLower.includes('ocr')) step = 'ocr';
  else if (stageLower.includes('formula')) step = 'formula';
  else if (stageLower.includes('ai format')) step = 'ai';
  else if (stageLower.includes('docx')) step = 'docx';

  if (typeof progress === 'number') updateProgress(progress, message);
  else if (message) $('progress-label').textContent = message;

  if (step) activateStep(step);

  if (status === 'done') {
    STEP_ORDER.forEach(s => markStepDone(s));
    updateProgress(100, 'Complete!');
    setStepsLineWidth(100);
    state.sseSource?.close();

    state.files.forEach(f => {
      if (f.status !== 'error') {
        f.status = 'done';
        updateQueueItemStatus(f.id, 'done');
      }
    });

    const btn = $('process-btn');
    if (btn) btn.disabled = false;
    $('merge-btn')?.classList.remove('hidden');

    loadPreview(state.currentJobId);
    showToast('Processing complete! 🎉', 'success', 4000);
  }

  if (status === 'error') {
    showToast(message || 'Processing error', 'error');
    state.sseSource?.close();
    const btn = $('process-btn');
    if (btn) btn.disabled = false;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   PROGRESS UI
   ═══════════════════════════════════════════════════════════════════════════ */
function showProgressArea(show) {
  const el = $('progress-area');
  if (!el) return;
  if (show) el.classList.add('visible');
  else      el.classList.remove('visible');
}

/** @param {number} pct  @param {string} label */
function updateProgress(pct, label) {
  const bar   = $('progress-bar');
  const pctEl = $('progress-pct');
  const lblEl = $('progress-label');
  if (bar)   bar.style.width = Math.min(100, pct) + '%';
  if (pctEl) pctEl.textContent = Math.round(pct) + '%';
  if (lblEl && label) lblEl.textContent = label;
}

/** @param {string} stepName */
function activateStep(stepName) {
  const idx = STEP_ORDER.indexOf(stepName);
  if (idx < 0) return;

  // Mark all previous steps as done
  STEP_ORDER.slice(0, idx).forEach(s => markStepDone(s));

  // Activate current step
  const stepEl = $(`step-${stepName}`);
  if (stepEl) {
    const circle = stepEl.querySelector('.step-circle');
    const icon   = stepEl.querySelector('.material-symbols-outlined');
    const label  = stepEl.querySelector('.step-label');
    if (circle) { circle.classList.add('active'); circle.classList.remove('done'); }
    if (icon)   icon.style.color = 'var(--primary)';
    if (label)  label.style.color = 'var(--primary)';
    stepEl.classList.add('active');
    stepEl.classList.remove('done');
    circle?.classList.add('step-active');
  }

  // Advance the connector line
  setStepsLineWidth((idx / (STEP_ORDER.length - 1)) * 100);
}

/** @param {string} stepName */
function markStepDone(stepName) {
  const stepEl = $(`step-${stepName}`);
  if (!stepEl) return;

  stepEl.classList.remove('active');
  stepEl.classList.add('done');

  const circle = stepEl.querySelector('.step-circle');
  const icon   = stepEl.querySelector('.material-symbols-outlined');
  const label  = stepEl.querySelector('.step-label');

  if (circle) {
    circle.classList.remove('step-active', 'active');
    circle.classList.add('done');
  }
  if (icon) {
    icon.textContent = 'check';
    icon.style.color = '#4ade80';
  }
  if (label) label.style.color = 'rgba(74,222,128,.70)';
}

function resetSteps() {
  STEP_ORDER.forEach(name => {
    const stepEl = $(`step-${name}`);
    if (!stepEl) return;
    stepEl.classList.remove('active', 'done');

    const circle = stepEl.querySelector('.step-circle');
    const icon   = stepEl.querySelector('.material-symbols-outlined');
    const label  = stepEl.querySelector('.step-label');

    if (circle) {
      circle.classList.remove('active', 'done', 'step-active');
      circle.style.borderColor = '';
    }
    if (icon)  { icon.textContent = STEP_ICONS[name]; icon.style.color = ''; }
    if (label) label.style.color = '';
  });
  setStepsLineWidth(0);
  updateProgress(0, 'Starting…');
}

/** @param {number} pct */
function setStepsLineWidth(pct) {
  const line = $('steps-line');
  if (line) line.style.width = pct + '%';
}

/* ═══════════════════════════════════════════════════════════════════════════
   PREVIEW  →  GET /preview/<job_id>
   ═══════════════════════════════════════════════════════════════════════════ */
async function loadPreview(jobId) {
  try {
    const res  = await fetch(`${API_BASE}/preview/${jobId}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Preview failed');

    state.pages       = data.pages || [];
    state.currentPage = 0;

    if (state.pages.length === 0) {
      showToast('No pages found in result', 'warn');
      return;
    }

    $('page-total').textContent = state.pages.length;
    showResultPanel();
    renderPage(0);

  } catch (err) {
    showToast('Preview load failed: ' + err.message, 'error');
  }
}

function showResultPanel() {
  $('empty-state')?.classList.add('hidden');
  $('result-panel')?.classList.remove('hidden');
}
function hideResultPanel() {
  $('empty-state')?.classList.remove('hidden');
  $('result-panel')?.classList.add('hidden');
}

/* ═══════════════════════════════════════════════════════════════════════════
   PAGE RENDERER
   ═══════════════════════════════════════════════════════════════════════════ */
/** @param {number} idx 0-indexed */
function renderPage(idx) {
  const page = state.pages[idx];
  if (!page) return;
  state.currentPage = idx;

  // Pagination controls
  $('page-current').textContent = idx + 1;
  const btnPrev = $('btn-prev');
  const btnNext = $('btn-next');
  if (btnPrev) btnPrev.disabled = idx === 0;
  if (btnNext) btnNext.disabled = idx === state.pages.length - 1;

  // Subject badge
  const subEl = $('subject-badge');
  if (subEl) subEl.textContent = page.subject || 'General';

  // Confidence badge
  const conf    = parseFloat(page.confidence) || 0;
  const confEl  = $('conf-badge');
  if (confEl) {
    confEl.textContent = `${Math.round(conf)}%`;
    confEl.className   = 'px-3 py-1.5 rounded-full border text-xs font-bold ' +
      (conf >= 80 ? 'conf-high' : conf >= 60 ? 'conf-mid' : 'conf-low');
  }

  // Title
  const titleEl = $('page-title');
  if (titleEl) titleEl.textContent = page.formatted?.title || page.label || `Page ${idx + 1}`;

  // Summary
  const summary   = page.formatted?.summary || '';
  const summaryEl = $('page-summary');
  if (summaryEl) {
    summaryEl.textContent = summary;
    summaryEl.classList.toggle('hidden', !summary);
  }

  // Key terms
  const terms      = page.formatted?.key_terms || [];
  const termsWrap  = $('key-terms-wrap');
  const termsEl    = $('key-terms');
  if (termsEl) termsEl.innerHTML = '';
  if (terms.length && termsEl) {
    terms.slice(0, 12).forEach(t => {
      const pill = document.createElement('span');
      pill.className = 'key-term-pill';
      pill.textContent = t;
      termsEl.appendChild(pill);
    });
    termsWrap?.classList.remove('hidden');
  } else {
    termsWrap?.classList.add('hidden');
  }

  // Sections
  renderSections(page);
}

/** @param {object} page */
function renderSections(page) {
  const container = $('sections-container');
  if (!container) return;
  container.innerHTML = '';

  const editable = state.editMode;
  const sections = page.formatted?.sections || [];

  if (sections.length === 0) {
    // Fallback: raw text
    const el = document.createElement('div');
    el.className = 'section-card glass-panel';
    el.innerHTML = `<p class="section-body" ${editable ? 'contenteditable="true"' : ''}>${escHtml(page.raw_text || 'No content')}</p>`;
    container.appendChild(el);
    return;
  }

  sections.forEach((sec) => {
    const el = document.createElement('div');
    el.className = 'section-card glass-panel' + (editable ? ' edit-mode-active' : '');

    let html = '';

    if (sec.heading) {
      html += `<div class="section-heading" ${editable ? 'contenteditable="true"' : ''}>${escHtml(sec.heading)}</div>`;
    }

    if (sec.content) {
      html += `<p class="section-body" ${editable ? 'contenteditable="true"' : ''}>${escHtml(sec.content)}</p>`;
    }

    if (sec.bullets?.length) {
      html += '<ul class="section-bullets">';
      sec.bullets.forEach(b => {
        html += `<li><span ${editable ? 'contenteditable="true"' : ''}>${escHtml(b)}</span></li>`;
      });
      html += '</ul>';
    }

    if (sec.formulas?.length) {
      sec.formulas.forEach(f => {
        html += `<div class="formula-block" ${editable ? 'contenteditable="true"' : ''}>${escHtml(f)}</div>`;
      });
    }

    if (sec.notes) {
      html += `<p class="section-note">${escHtml(sec.notes)}</p>`;
    }

    el.innerHTML = html;
    container.appendChild(el);
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   PAGE NAVIGATION
   ═══════════════════════════════════════════════════════════════════════════ */
function prevPage() {
  if (state.currentPage > 0) renderPage(state.currentPage - 1);
}

function nextPage() {
  if (state.currentPage < state.pages.length - 1) renderPage(state.currentPage + 1);
}

/* ═══════════════════════════════════════════════════════════════════════════
   EDIT MODE
   ═══════════════════════════════════════════════════════════════════════════ */
function toggleEditMode() {
  state.editMode = !state.editMode;

  const btn = $('edit-btn');
  if (btn) {
    btn.classList.toggle('active', state.editMode);
    const label = btn.querySelector('span:last-child');
    if (label) label.textContent = state.editMode ? 'Save' : 'Edit';
  }

  renderPage(state.currentPage);

  if (state.editMode) showToast('Edit mode on — click any text to edit', 'info');
  else                showToast('Preview updated', 'success');
}

/* ═══════════════════════════════════════════════════════════════════════════
   RE-PROCESS  →  POST /reprocess/<job_id>
   ═══════════════════════════════════════════════════════════════════════════ */
function reprocessPage() {
  $('rescan-overlay')?.classList.remove('hidden');
}

/** @param {'auto'|'light'|'scan'} mode */
async function doReprocess(mode) {
  $('rescan-overlay')?.classList.add('hidden');
  if (!state.currentJobId) return;

  const overrideSettings = { ...state.settings, enhance_mode: mode };

  try {
    const res  = await fetch(`${API_BASE}/reprocess/${state.currentJobId}`, {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({
        page_index: state.currentPage,
        settings:   overrideSettings,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Reprocess failed');

    showToast(`Re-scanning with "${mode}" mode…`, 'info');
    showProgressArea(true);
    resetSteps();
    connectSSE(state.currentJobId);

  } catch (err) {
    showToast(err.message, 'error');
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   DOWNLOAD  →  GET /download/<job_id>  &  GET /download-pdf/<job_id>
   ═══════════════════════════════════════════════════════════════════════════ */
function downloadDocx() {
  if (!state.currentJobId) { showToast('No document to download', 'warn'); return; }
  triggerDownload(`${API_BASE}/download/${state.currentJobId}`);
}

function downloadPdf() {
  if (!state.currentJobId) { showToast('No document to download', 'warn'); return; }
  triggerDownload(`${API_BASE}/download-pdf/${state.currentJobId}`);
}

/** @param {string} url */
function triggerDownload(url) {
  const a = document.createElement('a');
  a.href   = url;
  a.target = '_blank';
  a.rel    = 'noopener noreferrer';
  a.click();
}

/* ═══════════════════════════════════════════════════════════════════════════
   HISTORY  →  GET /history
   ═══════════════════════════════════════════════════════════════════════════ */
function openHistory() {
  const modal = $('history-modal');
  modal?.classList.add('open');
  fetchHistory();
}

function closeHistory() {
  $('history-modal')?.classList.remove('open');
}

async function fetchHistory() {
  const list = $('history-list');
  if (!list) return;
  list.innerHTML = '<div style="text-align:center;padding:2rem;color:rgba(172,170,177,.4);font-size:.875rem;">Loading…</div>';

  try {
    const res  = await fetch(`${API_BASE}/history`);
    const data = await res.json();
    state.historyData = data.sessions || [];
    renderHistoryList(list, state.historyData);

  } catch (err) {
    list.innerHTML = `<div style="text-align:center;padding:2rem;color:var(--error);font-size:.875rem;">Failed to load history.</div>`;
  }
}

/** @param {HTMLElement} list  @param {Array<object>} sessions */
function renderHistoryList(list, sessions) {
  list.innerHTML = '';
  if (sessions.length === 0) {
    list.innerHTML = '<div style="text-align:center;padding:2rem;color:rgba(172,170,177,.4);font-size:.875rem;">No sessions yet.</div>';
    return;
  }
  sessions.forEach(s => {
    const el  = document.createElement('div');
    el.className = 'history-item glass-panel';
    const pgCount = (s.pages || []).length;
    el.innerHTML = `
      <div style="min-width:0;flex:1;">
        <p style="font-size:.875rem;font-weight:700;color:var(--on-surface);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escHtml(s.job_id || 'Session')}</p>
        <p style="font-size:.75rem;color:var(--on-surface-variant);">${pgCount} page${pgCount !== 1 ? 's' : ''} · ${formatDate(s.created_at)}</p>
      </div>
      <div style="display:flex;gap:.5rem;flex-shrink:0;">
        <button onclick="loadHistorySession('${escHtml(s.job_id)}')" class="dl-btn-docx" style="font-size:.75rem;padding:.375rem .75rem;">Load</button>
        <button onclick="triggerDownload('${API_BASE}/download/${escHtml(s.job_id)}')" class="dl-btn-pdf" style="font-size:.75rem;padding:.375rem .75rem;">DOCX</button>
      </div>`;
    list.appendChild(el);
  });
}

/** @param {string} jobId */
async function loadHistorySession(jobId) {
  closeHistory();
  state.currentJobId = jobId;
  showProgressArea(false);
  showToast('Loading session…', 'info');
  await loadPreview(jobId);
}

/* ═══════════════════════════════════════════════════════════════════════════
   MERGE  →  POST /merge
   ═══════════════════════════════════════════════════════════════════════════ */
function openMergeModal() {
  $('merge-modal')?.classList.add('open');
  state.mergeIds = [];
  fetchMergeList();
}

function closeMergeModal() {
  $('merge-modal')?.classList.remove('open');
}

async function fetchMergeList() {
  const list = $('merge-list');
  if (!list) return;
  list.innerHTML = '<div style="text-align:center;padding:1rem;color:rgba(172,170,177,.4);font-size:.875rem;">Loading sessions…</div>';

  try {
    const res      = await fetch(`${API_BASE}/history`);
    const data     = await res.json();
    const sessions = data.sessions || [];
    list.innerHTML = '';

    if (sessions.length === 0) {
      list.innerHTML = '<div style="text-align:center;padding:1rem;color:rgba(172,170,177,.4);font-size:.875rem;">No sessions available.</div>';
      return;
    }

    sessions.forEach(s => {
      const label = document.createElement('label');
      label.className = 'merge-item glass-panel';
      const pgCount = (s.pages || []).length;
      label.innerHTML = `
        <input type="checkbox" value="${escHtml(s.job_id)}"
          onchange="toggleMergeId('${escHtml(s.job_id)}', this.checked)"
          style="width:1rem;height:1rem;accent-color:var(--tertiary);flex-shrink:0;cursor:pointer;"/>
        <div>
          <p style="font-size:.875rem;font-weight:700;color:var(--on-surface);">${escHtml(s.job_id)}</p>
          <p style="font-size:.75rem;color:var(--on-surface-variant);">${pgCount} page${pgCount !== 1 ? 's' : ''}</p>
        </div>`;
      list.appendChild(label);
    });

  } catch {
    list.innerHTML = '<div style="text-align:center;padding:1rem;color:var(--error);font-size:.875rem;">Failed to load sessions.</div>';
  }
}

/** @param {string} id  @param {boolean} checked */
function toggleMergeId(id, checked) {
  if (checked) state.mergeIds.push(id);
  else         state.mergeIds = state.mergeIds.filter(x => x !== id);
}

async function doMerge() {
  if (state.mergeIds.length < 2) {
    showToast('Select at least 2 sessions to merge', 'warn');
    return;
  }
  const btn = document.querySelector('[onclick="doMerge()"]');
  if (btn) { btn.disabled = true; btn.textContent = 'Merging…'; }

  try {
    const res  = await fetch(`${API_BASE}/merge`, {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ job_ids: state.mergeIds, settings: state.settings }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Merge failed');

    closeMergeModal();
    triggerDownload(`${API_BASE}/download/${data.merged_job_id}`);
    showToast('Merged document ready! 📄', 'success');

  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Merge & Download'; }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   CAMERA CAPTURE
   ═══════════════════════════════════════════════════════════════════════════ */
async function openCamera() {
  $('camera-modal')?.classList.add('open');
  try {
    state.cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
    });
    const video = $('camera-video');
    if (video) video.srcObject = state.cameraStream;

  } catch (err) {
    showToast('Camera access denied. Check browser permissions.', 'error');
    $('camera-modal')?.classList.remove('open');
  }
}

function closeCamera() {
  $('camera-modal')?.classList.remove('open');
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }
  const video = $('camera-video');
  if (video) video.srcObject = null;
}

function capturePhoto() {
  const video  = $('camera-video');
  const canvas = $('camera-canvas');
  if (!video || !canvas) return;

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  canvas.toBlob((blob) => {
    if (!blob) { showToast('Capture failed — try again', 'error'); return; }
    const file = new File([blob], `capture-${Date.now()}.jpg`, { type: 'image/jpeg' });
    addFileToQueue(file);
    closeCamera();
    showToast('Photo captured!', 'success');
  }, 'image/jpeg', 0.95);
}

/* ═══════════════════════════════════════════════════════════════════════════
   NAV SCROLL EFFECT
   ═══════════════════════════════════════════════════════════════════════════ */
function initNavScroll() {
  const nav = document.querySelector('.nav-bar');
  if (!nav) return;
  const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 40);
  window.addEventListener('scroll', onScroll, { passive: true });
}

/* ═══════════════════════════════════════════════════════════════════════════
   DRAG & DROP EVENTS (drop zone)
   ═══════════════════════════════════════════════════════════════════════════ */
function initDropZone() {
  const zone = $('drop-zone');
  if (!zone) return;

  zone.addEventListener('dragover',  (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', ()  => zone.classList.remove('drag-over'));
  zone.addEventListener('drop',      (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    Array.from(e.dataTransfer.files).forEach(addFileToQueue);
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   KEYBOARD SHORTCUTS
   ═══════════════════════════════════════════════════════════════════════════ */
function initKeyboard() {
  document.addEventListener('keydown', (e) => {
    const appOpen = $('app-overlay')?.classList.contains('open');

    // Escape — close any open overlay/modal
    if (e.key === 'Escape') {
      if (!appOpen) return;
      $('rescan-overlay')?.classList.add('hidden');
      closeCamera();
      closeHistory();
      closeMergeModal();
      return;
    }

    // Arrow keys for page navigation (only in app, not in edit mode)
    if (appOpen && !state.editMode) {
      if (e.key === 'ArrowLeft')  prevPage();
      if (e.key === 'ArrowRight') nextPage();
    }

    // Ctrl/Cmd + E → toggle edit mode
    if (appOpen && (e.ctrlKey || e.metaKey) && e.key === 'e') {
      e.preventDefault();
      toggleEditMode();
    }

    // Ctrl/Cmd + D → download DOCX
    if (appOpen && (e.ctrlKey || e.metaKey) && e.key === 'd') {
      e.preventDefault();
      downloadDocx();
    }
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   SMOOTH SCROLL ANCHORS
   ═══════════════════════════════════════════════════════════════════════════ */
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', (e) => {
      const target = document.querySelector(a.getAttribute('href'));
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   INTERSECTION OBSERVER — reveal on scroll
   ═══════════════════════════════════════════════════════════════════════════ */
function initScrollReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('fade-up');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  document.querySelectorAll('.reveal-on-scroll').forEach(el => observer.observe(el));
}

/* ═══════════════════════════════════════════════════════════════════════════
   ACTIVE SETTING BUTTON INIT
   ═══════════════════════════════════════════════════════════════════════════ */
function initSettingButtons() {
  // Mark default active states
  $$('.theme-btn').forEach(btn => {
    btn.classList.toggle('active-theme', btn.dataset.theme === state.settings.theme);
  });
  $$('.mode-btn').forEach(btn => {
    btn.classList.toggle('active-mode', btn.dataset.mode === state.settings.enhance_mode);
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   HEALTH CHECK — verify Flask API is reachable
   ═══════════════════════════════════════════════════════════════════════════ */
async function checkAPIHealth() {
  try {
    const res = await fetch(`${API_BASE}/`, { method: 'GET', signal: AbortSignal.timeout(3000) });
    if (!res.ok) console.warn('[AI Notes] Flask API returned', res.status);
  } catch {
    console.warn('[AI Notes] Flask API not reachable — is the server running?');
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initNavScroll();
  initDropZone();
  initKeyboard();
  initSmoothScroll();
  initScrollReveal();
  initSettingButtons();
  checkAPIHealth();
});

/* ═══════════════════════════════════════════════════════════════════════════
   GLOBAL FUNCTION EXPORTS
   (These are called by inline onclick= handlers in index.html)
   ═══════════════════════════════════════════════════════════════════════════ */
window.openApp          = openApp;
window.closeApp         = closeApp;
window.toggleSettings   = toggleSettings;
window.setSetting       = setSetting;
window.handleFileSelect = handleFileSelect;
window.handleDrop       = handleDrop;
window.removeFromQueue  = removeFromQueue;
window.clearQueue       = clearQueue;
window.startProcessing  = startProcessing;
window.prevPage         = prevPage;
window.nextPage         = nextPage;
window.toggleEditMode   = toggleEditMode;
window.reprocessPage    = reprocessPage;
window.doReprocess      = doReprocess;
window.downloadDocx     = downloadDocx;
window.downloadPdf      = downloadPdf;
window.triggerDownload  = triggerDownload;
window.openHistory      = openHistory;
window.closeHistory     = closeHistory;
window.loadHistorySession = loadHistorySession;
window.openMergeModal   = openMergeModal;
window.closeMergeModal  = closeMergeModal;
window.toggleMergeId    = toggleMergeId;
window.doMerge          = doMerge;
window.openCamera       = openCamera;
window.closeCamera      = closeCamera;
window.capturePhoto     = capturePhoto;
