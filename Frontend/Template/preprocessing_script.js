/*  <script src="preprocessing_script.js?v=4"></script>
imized for maximum speed + live undo/redo/reset */
var API = 'http://127.0.0.1:8000/api';
var SID = null;
var numCols = [], catCols = [], allCols = [];
var numColsWithNull = [], catColsWithNull = [];  // only cols that have nulls
var encodedCols = new Set();
var scaledCols = new Set();      // columns that have been scaled (front-end tracking)
var _scalingStack = [];          // undo stack: each entry = array of cols scaled in that batch
var _scalingRedoStack = [];      // redo stack: mirrors undo for redo-of-scaling ops
var origShape = null;
var _plotlyLoaded = false;
var _pickers = {};
var _activeSection = 'missing';  // track which section is currently visible

/* ── Info texts ─────────────────────────────── */
var INFO = {
    missing: { t: 'Missing Values', b: 'Missing values are empty cells. Most ML models fail when they encounter them.<br><br><b>Numerical:</b> Fill with Mean, Median, a custom number, or drop rows with missing data.<br><b>Categorical:</b> Fill with Mode (most common), a custom word, or drop the row.<br><br>If no columns are selected, the operation applies to every column of that type.' },
    duplicates: { t: 'Duplicate Rows', b: 'Duplicate rows are identical across every column. They add no new information and bias model training.<br><br>You can keep the first occurrence or the last one when removing.<br><br><b>Note:</b> Some preprocessing steps (like filling nulls with the same value) can <em>create</em> new duplicates — that\'s why the duplicate count updates after every operation.' },
    constant: { t: 'Constant Columns', b: 'A constant column has the same value in every single row (e.g. "Country" = "India" for all 10,000 rows). It provides zero information to a model and wastes memory. Always remove before training.' },
    encoding: { t: 'Categorical Encoding', b: 'ML models only understand numbers. All text/category columns must be converted first.<br><br><b>Label:</b> Red=0, Blue=1, Green=2. Simple, fast. Good for tree-based models.<br><b>One-Hot:</b> Creates a new 0/1 column for each category. Best for unordered text.<br><b>Ordinal:</b> You define the order: Low=0, Medium=1, High=2.<br><b>Target:</b> Replaces each category with the average target value for that group. Powerful but prone to data leakage.' },
    scaling: { t: 'Feature Scaling', b: 'Scaling puts all numerical columns on the same scale so columns with larger numbers do not dominate the model.<br><br><b>Standard:</b> Mean=0, Std=1. Works for most algorithms.<br><b>Min-Max:</b> Values become 0 to 1. Good for bounded features.<br><b>Robust:</b> Based on median, best when data has outliers.<br><br>Required for: KNN, SVM, Neural Networks, PCA.' },
    outliers: { t: 'Outlier Handling', b: 'Outliers are extreme values far from the rest. They distort model training.<br><br><b>Detection methods:</b><br>IQR: Outside Q1-1.5*IQR or Q3+1.5*IQR.<br>Z-Score: More than 3 standard deviations from mean.<br><br><b>Actions:</b><br>Remove: delete entire row.<br>Cap: clip value to the boundary.<br>Flag: add a 0/1 "is_outlier" column and keep the row.' },
    sampling: { t: 'Class Balancing', b: 'If your dataset has 95% "No" and 5% "Yes", a model can predict "No" for everything and get 95% accuracy — but it is completely useless.<br><br><b>SMOTE:</b> Creates new synthetic minority-class rows. Best option.<br><b>Over-sample:</b> Duplicates existing minority rows.<br><b>Under-sample:</b> Removes majority class rows. Reduces dataset size.' }
};

function showInfo(k) { var d = INFO[k]; if (!d) return; document.getElementById('modalT').textContent = d.t; document.getElementById('modalB').innerHTML = d.b; document.getElementById('modal').classList.add('show'); }
function closeModal() { document.getElementById('modal').classList.remove('show'); }

/* ── Toast ───────────────────────────────────── */
var _tt = null;
function toast(msg, type) {
    var t = document.getElementById('toast');
    t.textContent = msg; t.className = 'toast show ' + (type || 'tok');
    clearTimeout(_tt); _tt = setTimeout(function () { t.classList.remove('show'); }, 2800);
}

/* ── Sidebar ─────────────────────────────────── */
function toggleSB() { document.getElementById('sidebar').classList.toggle('open'); document.getElementById('ov').style.display = 'block'; }
function closeSB() { document.getElementById('sidebar').classList.remove('open'); document.getElementById('ov').style.display = 'none'; }
function go(name) {
    document.querySelectorAll('.sec').forEach(function (s) { s.style.display = 'none'; });
    var el = document.getElementById('sec-' + name); if (el) el.style.display = '';
    document.querySelectorAll('.nav-btn').forEach(function (b) { b.classList.remove('active'); if (b.getAttribute('data-s') === name) b.classList.add('active'); });
    _activeSection = name;
    closeSB();
    if (name === 'missing') loadMissing();
    else if (name === 'duplicates') loadDups();
    else if (name === 'constant') loadConst();
    else if (name === 'report') loadReport();
}
function markDone(n) {
    var el = document.getElementById('n' + n);
    if (el) { el.textContent = '\u2713'; el.parentElement.classList.add('done'); }
}
function markUndone(n) {
    var el = document.getElementById('n' + n);
    if (el) { el.textContent = n; el.parentElement.classList.remove('done'); }
}

/* ── Option card toggle (radio visual) ───────── */
function optPick(card, groupId) {
    var r = card.querySelector('input[type=radio]'); if (r) r.checked = true;
    var g = document.getElementById(groupId) || card.closest('.opt-row'); if (!g) return;
    g.querySelectorAll('.opt-c').forEach(function (c) { c.classList.remove('on'); });
    card.classList.add('on');
}
document.querySelectorAll('.opt-c input[type=radio]:checked').forEach(function (r) { r.closest('.opt-c').classList.add('on'); });
function tgPick(label) {
    var r = label.querySelector('input[type=radio]'); if (r) r.checked = true;
    var row = label.closest('.tg-row'); if (!row) return;
    row.querySelectorAll('.tg-opt').forEach(function (t) { t.classList.remove('on'); });
    label.classList.add('on');
}

function toggleOrdM() { document.getElementById('ordM').style.display = document.getElementById('ordA').checked ? 'none' : ''; }

/* ── Column Picker ───────────────────────────── */
function buildPicker(id, cols, type, excludeSet) {
    var el = document.getElementById('pk-' + id); if (!el) return;
    var show = excludeSet ? cols.filter(function (c) { return !excludeSet.has(c); }) : cols;
    _pickers[id] = { cols: show, type: type, selected: new Set() };
    el.innerHTML = '';
    if (!show || !show.length) {
        var msg = (excludeSet && cols.length) ? 'All columns have already been encoded.' : 'No columns available.';
        el.innerHTML = '<div style="padding:10px 12px;font-size:12px;color:#94a3b8;font-style:italic">' + msg + '</div>'; return;
    }
    var bar = document.createElement('div'); bar.className = 'cpicker-bar';
    var srch = document.createElement('input'); srch.className = 'cpicker-search'; srch.type = 'text'; srch.placeholder = 'Search ' + show.length + ' columns...'; srch.autocomplete = 'off';
    srch.oninput = function () { _filterPicker(id, this.value); };
    var btnAll = _mkBarBtn('All', function () { _setAll(id, true); });
    var btnNone = _mkBarBtn('None', function () { _setAll(id, false); });
    bar.appendChild(srch); bar.appendChild(btnAll); bar.appendChild(btnNone);
    var list = document.createElement('div'); list.className = 'cpicker-list'; list.id = 'pkl-' + id;
    // Use DocumentFragment for batch DOM insert (much faster)
    var frag = document.createDocumentFragment();
    show.forEach(function (col) { frag.appendChild(_makeItem(id, col, type)); });
    list.appendChild(frag);
    var foot = document.createElement('div'); foot.className = 'cpicker-foot';
    var sel = document.createElement('span'); sel.id = 'pkf-' + id; sel.className = 'auto-hint'; sel.textContent = 'All ' + show.length + ' columns (no selection = apply to all)';
    foot.appendChild(sel);
    el.appendChild(bar); el.appendChild(list); el.appendChild(foot);
}

function _mkBarBtn(label, fn) {
    var b = document.createElement('button'); b.className = 'cpicker-btn'; b.textContent = label; b.onclick = fn; return b;
}
function _makeItem(pickerId, col, type) {
    var item = document.createElement('div'); item.className = 'cpicker-item'; item.setAttribute('data-col', col);
    var chk = document.createElement('div'); chk.className = 'cpicker-check';
    var name = document.createElement('div'); name.className = 'cpicker-name'; name.textContent = col; name.title = col;
    var badge = document.createElement('span'); badge.className = 'cpicker-type ' + (type === 'num' ? 'type-num' : 'type-cat'); badge.textContent = type === 'num' ? 'NUM' : 'CAT';
    item.appendChild(chk); item.appendChild(name); item.appendChild(badge);
    item.onclick = function () { _toggleItem(pickerId, col, item); };
    return item;
}
function _toggleItem(pickerId, col, item) {
    var st = _pickers[pickerId]; if (!st) return;
    if (st.selected.has(col)) { st.selected.delete(col); item.classList.remove('sel'); }
    else { st.selected.add(col); item.classList.add('sel'); }
    _updateFoot(pickerId);
}
function _setAll(pickerId, select) {
    var st = _pickers[pickerId]; if (!st) return;
    if (select) st.cols.forEach(function (c) { st.selected.add(c); });
    else st.selected.clear();
    var list = document.getElementById('pkl-' + pickerId); if (!list) return;
    list.querySelectorAll('.cpicker-item').forEach(function (item) {
        var col = item.getAttribute('data-col');
        if (item.style.display === 'none') return;
        if (st.selected.has(col)) item.classList.add('sel'); else item.classList.remove('sel');
    });
    _updateFoot(pickerId);
}
function _filterPicker(pickerId, query) {
    var list = document.getElementById('pkl-' + pickerId); if (!list) return;
    var q = query.toLowerCase().trim();
    list.querySelectorAll('.cpicker-item').forEach(function (item) {
        var col = item.getAttribute('data-col');
        item.style.display = (!q || col.toLowerCase().includes(q)) ? 'flex' : 'none';
    });
}
function _updateFoot(pickerId) {
    var st = _pickers[pickerId]; if (!st) return;
    var foot = document.getElementById('pkf-' + pickerId); if (!foot) return;
    var n = st.selected.size, total = st.cols.length;
    foot.textContent = n === 0 ? 'All ' + total + ' columns (no selection = apply to all)' : n + ' of ' + total + ' selected';
}
function getPicked(pickerId) {
    var st = _pickers[pickerId]; if (!st) return [];
    return Array.from(st.selected);
}

/* ── Button loading state helpers ────────────── */
function setBtnLoading(btnId, loading) {
    var btn = document.getElementById(btnId); if (!btn) return;
    if (loading) { btn._orig = btn.textContent; btn.disabled = true; btn.textContent = '⏳ Working...'; }
    else { btn.disabled = false; btn.textContent = btn._orig || btn.textContent; }
}

/* ── API helper ──────────────────────────────── */
async function req(method, path, body) {
    var opts = { method: method, headers: {} };
    if (body) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
    var r = await fetch(API + path, opts);
    if (!r.ok) { var e = await r.json().catch(function () { return { detail: 'Error' }; }); throw new Error(e.detail || 'Request failed'); }
    return r.json();
}
function popSel(id, cols, placeholder) {
    var el = document.getElementById(id); if (!el) return;
    // Build via innerHTML for speed
    var html = placeholder ? '<option value="">' + placeholder + '</option>' : '';
    html += (cols || []).map(function (c) { return '<option value="' + c + '">' + c + '</option>'; }).join('');
    el.innerHTML = html;
}

/* ── Init ────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', function () {
    var p = new URLSearchParams(window.location.search);
    SID = p.get('session_id') || localStorage.getItem('session_id');
    if (!SID) { window.location.href = 'upload.html'; return; }
    if (window.innerWidth < 660) document.getElementById('menuBtn').style.display = 'block';
    // Fire both in parallel, then load missing values once both finish
    Promise.all([loadStats(true), loadCols()]).then(function () {
        loadMissing();
        loadHist(); // load history once, separately
        _refreshBadges(); // initial three-badge load
    });
});

/* ── Stats (no auto-history reload) ─────────── */
async function loadStats(skipHist) {
    try {
        var d = await req('GET', '/preprocessing/' + SID + '/stats');
        if (!origShape) origShape = { rows: d.original_rows || d.current_rows, cols: d.original_columns || d.current_columns };
        document.getElementById('dsN').textContent = d.dataset_name || 'Dataset';
        document.getElementById('dsS').textContent = d.current_rows + ' rows, ' + d.current_columns + ' cols';
        document.getElementById('undoB').disabled = !d.can_undo;
        document.getElementById('redoB').disabled = !d.can_redo;
        if (!skipHist) loadHist();
        return d;
    } catch (e) { document.getElementById('dsN').textContent = 'Dataset'; document.getElementById('dsS').textContent = 'Server not connected'; }
}

/* ── FAST column load — uses lightweight endpoint ── */
async function loadCols() {
    try {
        // Use the new fast-columns endpoint — no EDA computation, just column names/types
        var d = await req('GET', '/eda/' + SID + '/fast-columns');
        if (d.statistics) {
            allCols = d.statistics.columns || [];
            numCols = d.statistics.numerical_columns || [];
            catCols = d.statistics.categorical_columns || [];
            _rebuildPickers();
        }
    } catch (e) { /* silent */ }
}

function _rebuildPickers() {
    // Missing value pickers: only show columns that actually have nulls
    _buildMissingPicker('numMis', numColsWithNull, 'num');
    _buildMissingPicker('catMis', catColsWithNull, 'cat');
    // Encoding / scaling / outlier pickers use all columns as before
    buildPicker('label', catCols, 'cat', encodedCols);
    buildPicker('onehot', catCols, 'cat', encodedCols);
    buildPicker('target', catCols, 'cat', encodedCols);
    buildPicker('scale', numCols, 'num');
    buildPicker('out', numCols, 'num');
    var unencCat = catCols.filter(function (c) { return !encodedCols.has(c); });
    popSel('ordC', unencCat, 'Select column...');
    popSel('tgtC', allCols, 'Select target...');
    popSel('sampT', catCols, 'Select target column...');
}

/* ── Sync encoded-column state from backend ─────────────────────────────────
   After undo/redo/reset the backend DataFrame is in a new state.
   The safest way to re-derive which columns are encoded is:
     1. Clear encodedCols (we don’t know what changed)
     2. Re-fetch column lists from the live DataFrame via loadCols()
        — columns that were label/ordinal encoded are now int64 so they
          appear in numCols, not catCols; after undo they are object again
          and re-appear in catCols automatically.
     3. Use the backend’s /encodable-columns list to mark any catCols that
        are STILL encoded (e.g. after partial undo) back into encodedCols.
   This is called exclusively from _fullRefresh() after undo/redo/reset.   */
async function _syncEncodedCols() {
    try {
        /* First: reload column lists so catCols reflects the current df dtypes */
        await loadCols();
        /* Second: ask backend which catCols are still original_categorical         */
        var d = await req('GET', '/preprocessing/' + SID + '/encodable-columns');
        var encodable = new Set(d.encodable_columns || []);
        /* Any catCol NOT in encodable is still encoded (rare after undo, but safe) */
        encodedCols.clear();
        catCols.forEach(function (c) {
            if (!encodable.has(c)) encodedCols.add(c);
        });
        /* Rebuild every picker/dropdown that depends on encodedCols */
        buildPicker('label', catCols, 'cat', encodedCols);
        buildPicker('onehot', catCols, 'cat', encodedCols);
        buildPicker('target', catCols, 'cat', encodedCols);
        var unenc = catCols.filter(function (c) { return !encodedCols.has(c); });
        popSel('ordC', unenc, 'Select column...');
    } catch (e) { /* silent — non-critical */ }
}

/* Build the missing-value picker — shows only null columns or a clean message */
function _buildMissingPicker(pickerId, nullCols, type) {
    var el = document.getElementById('pk-' + pickerId);
    var ctrlDiv = document.getElementById(pickerId + '-controls');

    if (!nullCols || nullCols.length === 0) {
        /* No nulls — show green success message, hide entire controls section */
        if (el) {
            el.innerHTML =
                '<div style="padding:14px 16px;background:#f0fdf4;border:1.5px solid #bbf7d0;' +
                'border-radius:8px;display:flex;align-items:center;gap:10px;">' +
                '<span style="font-size:20px">\u2705</span>' +
                '<div><div style="font-size:13px;font-weight:700;color:#166534">No null values found</div>' +
                '<div style="font-size:11px;color:#15803d;margin-top:2px">' +
                (type === 'num' ? 'All numerical' : 'All categorical') +
                ' columns are complete \u2014 nothing to fix</div></div></div>';
        }
        /* Hide fill method + apply button completely */
        if (ctrlDiv) ctrlDiv.style.display = 'none';
        return;
    }

    /* Has nulls — build normal picker and show the controls */
    buildPicker(pickerId, nullCols, type);
    if (ctrlDiv) ctrlDiv.style.display = '';
}

async function loadHist() {
    try {
        var hist = await req('GET', '/preprocessing/' + SID + '/history');
        var el = document.getElementById('histL'); if (!el) return;
        if (!hist || !hist.length) { el.innerHTML = '<span style="color:#cbd5e1;font-style:italic;font-size:11px">None yet</span>'; return; }
        // Build string — faster than map+join on large arrays
        var html = '';
        var slice = hist.slice().reverse().slice(0, 25);
        for (var i = 0; i < slice.length; i++) {
            var h = slice[i];
            var t = new Date(h.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            var num = hist.length - i;
            html += '<div style="padding:4px 0;border-bottom:1px solid #f1f5f9;line-height:1.5">'
                + '<span style="font-weight:700;color:#6366f1;font-size:11px">' + num + '.</span> '
                + '<span style="font-size:11px;color:#374151">' + h.description + '</span><br>'
                + '<span style="color:#94a3b8;font-size:10px">' + t + '</span></div>';
        }
        el.innerHTML = html;
    } catch (e) { /* silent */ }
}

/* ── Live three-badge refresh ─────────────────────────────────────────
   Calls three existing endpoints in parallel.
   Each badge + each sidebar step marker is updated independently.
   A single API failure can never blank other badges.                 */
async function _refreshBadges() {
    var base = '/preprocessing/' + SID;

    /* ── 1. NULLS ── */
    req('GET', base + '/missing-values').then(function (d) {
        var n = d.total_null_count || 0;
        _paintBadge('badge-nulls', n === 0, 'Nulls: 0', 'Nulls: ' + n);
        /* Step marker: done only when ALL nulls (num + cat) are gone */
        if (n === 0) markDone(1); else markUndone(1);
    }).catch(function () { /* badge stays as-is */ });

    /* ── 2. DUPLICATES ── */
    req('GET', base + '/duplicates').then(function (d) {
        var remove = (typeof d.rows_to_remove !== 'undefined')
            ? d.rows_to_remove
            : Math.floor((d.duplicate_row_count || 0) / 2);
        _paintBadge('badge-dups', remove === 0,
            'Duplicates: 0',
            'Duplicates: ' + remove);
        if (remove === 0) markDone(2); else markUndone(2);
    }).catch(function () { /* badge stays as-is */ });

    /* ── 3. CONSTANT COLS ── */
    req('GET', base + '/constant-columns').then(function (d) {
        var c = (d.constant_columns || []).length;
        _paintBadge('badge-const', c === 0,
            'Constant Cols: 0',
            'Constant Cols: ' + c);
        if (c === 0) markDone(3); else markUndone(3);
    }).catch(function () { /* badge stays as-is */ });
}

/* Paint a single always-visible pill badge.
   isClean=true → green background, false → red background. */
function _paintBadge(id, isClean, labelClean, labelIssue) {
    var el = document.getElementById(id); if (!el) return;
    el.textContent = isClean ? labelClean : labelIssue;
    el.style.fontWeight = '700';
    el.style.background = isClean ? '#f0fdf4' : '#fff1f2';
    el.style.color = isClean ? '#166534' : '#be123c';
    el.style.border = isClean ? '1.5px solid #bbf7d0' : '1.5px solid #fca5a5';
}


/* ── Refresh the currently active section live ── */
async function _refreshActiveSection() {
    if (_activeSection === 'missing') {
        await loadMissing();
    } else if (_activeSection === 'duplicates') {
        await loadDups();
    } else if (_activeSection === 'constant') {
        await loadConst();
    } else if (_activeSection === 'report') {
        await loadReport();
    }
    /* For scaling/outliers/sampling, rebuild pickers to reflect new data.
       Encoding pickers are always rebuilt by _syncEncodedCols() above,
       so we skip loadCols() for encoding to avoid a redundant double-call. */
    if (_activeSection === 'scaling' || _activeSection === 'outliers' || _activeSection === 'sampling') {
        await loadCols();
    }
}

/* ── After-action refresh: stats + history + badge + active section ── */
async function _refresh() {
    return Promise.all([
        req('GET', '/preprocessing/' + SID + '/stats').then(function (d) {
            if (!origShape) origShape = { rows: d.original_rows || d.current_rows, cols: d.original_columns || d.current_columns };
            document.getElementById('dsN').textContent = d.dataset_name || 'Dataset';
            document.getElementById('dsS').textContent = d.current_rows + ' rows, ' + d.current_columns + ' cols';
            document.getElementById('undoB').disabled = !d.can_undo;
            document.getElementById('redoB').disabled = !d.can_redo;
        }).catch(function () { }),
        loadHist(),
        _refreshBadges()  // always update all three badges after every operation
    ]);
}

/* ── Full refresh after undo/redo/reset ────────────────────────────────────
   Clears encodedCols first so stale frontend state can’t bleed through,
   then re-syncs everything from the backend in the right order.           */
async function _fullRefresh() {
    await _refresh();
    /* Clear stale encoding state before re-syncing */
    encodedCols.clear();
    /* Re-sync: reloads catCols from live df, then derives encodedCols */
    await _syncEncodedCols();
    await _refreshActiveSection();
}

/* ── MISSING VALUES ──────────────────────────── */
async function loadMissing() {
    try {
        var d = await req('GET', '/preprocessing/' + SID + '/missing-values');
        document.getElementById('mv-t').textContent = d.total_null_count || 0;
        document.getElementById('mv-p').textContent = (d.total_null_percentage || 0) + '%';
        document.getElementById('mv-r').textContent = d.total_rows || '-';
        document.getElementById('mv-c').textContent = (d.columns || []).filter(function (c) { return c.null_count > 0; }).length;

        var tb = document.getElementById('mvB');
        if (!d.columns || !d.columns.length) {
            tb.innerHTML = '<tr><td colspan="5" style="padding:12px;text-align:center;color:#94a3b8">No columns</td></tr>';
            return;
        }
        tb.innerHTML = d.columns.map(function (c) {
            var lv = c.null_percentage > 50 ? ['br', 'High'] : c.null_percentage > 20 ? ['by', 'Med'] : c.null_percentage > 0 ? ['bgr', 'Low'] : ['bg', 'None'];
            return '<tr><td><b>' + c.column + '</b></td><td>' + c.data_type + '</td><td>' + c.null_count + '</td><td>' + c.null_percentage + '%</td><td><span class="b ' + lv[0] + '">' + lv[1] + '</span></td></tr>';
        }).join('');

        /* ── Classify columns by data_type from the API response directly.
               This avoids cross-contamination from the separately-loaded
               numCols / catCols arrays which may be stale or empty.        */
        function _isNumType(dtype) {
            if (!dtype) return false;
            var t = dtype.toLowerCase();
            return t.indexOf('int') !== -1 || t.indexOf('float') !== -1 ||
                t.indexOf('number') !== -1 || t.indexOf('decimal') !== -1 ||
                t.indexOf('double') !== -1 || t.indexOf('numeric') !== -1;
        }

        var nullCols = (d.columns || []).filter(function (c) { return c.null_count > 0; });

        /* Use data_type from the response to split — never touch the other picker */
        var numWithNull = nullCols
            .filter(function (c) { return _isNumType(c.data_type); })
            .map(function (c) { return c.column; });
        var catWithNull = nullCols
            .filter(function (c) { return !_isNumType(c.data_type); })
            .map(function (c) { return c.column; });

        /* Only rebuild pickers that actually changed — keeps other section stable */
        numColsWithNull = numWithNull;
        catColsWithNull = catWithNull;

        _buildMissingPicker('numMis', numColsWithNull, 'num');
        _buildMissingPicker('catMis', catColsWithNull, 'cat');

        if ((d.total_null_count || 0) === 0) markDone(1);
    } catch (e) { /* silent */ }
}


async function applyNM() {
    var picked = getPicked('numMis');
    var radio = document.querySelector('input[name="nMis"]:checked');
    if (!radio) { toast('Pick a fill method', 'terr'); return; }

    /* ALWAYS send explicit column list — never send null/undefined
       (sending null tells the backend to process ALL column types,
        which would cross-contaminate categorical columns).          */
    var targetCols = picked.length ? picked : numColsWithNull;
    if (!targetCols.length) { toast('No numerical columns with missing values', 'tinf'); return; }

    var s = radio.value;
    var body = { columns: targetCols, strategy: s };
    if (s === 'constant_num') body.constant_value = parseFloat(document.getElementById('cNum').value) || 0;

    setBtnLoading('applyNMBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/missing-values', body);
        toast('Numerical columns updated (' + targetCols.length + ' col' + (targetCols.length !== 1 ? 's' : '') + ')');
        await loadMissing();
        _refresh();
    } catch (e) { toast(e.message, 'terr'); }
    finally { setBtnLoading('applyNMBtn', false); }
}

async function applyCM() {
    var picked = getPicked('catMis');
    var radio = document.querySelector('input[name="cMis"]:checked');
    if (!radio) { toast('Pick a fill method', 'terr'); return; }

    /* ALWAYS send explicit column list — never send null/undefined */
    var targetCols = picked.length ? picked : catColsWithNull;
    if (!targetCols.length) { toast('No categorical columns with missing values', 'tinf'); return; }

    var s = radio.value;
    var body = { columns: targetCols, strategy: s };
    if (s === 'constant_cat') body.constant_string = document.getElementById('cCat').value || 'Unknown';

    setBtnLoading('applyCMBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/missing-values', body);
        toast('Categorical columns updated (' + targetCols.length + ' col' + (targetCols.length !== 1 ? 's' : '') + ')');
        await loadMissing();
        _refresh();
    } catch (e) { toast(e.message, 'terr'); }
    finally { setBtnLoading('applyCMBtn', false); }
}

/* ── DUPLICATES ──────────────────────────────── */
async function loadDups() {
    try {
        var d = await req('GET', '/preprocessing/' + SID + '/duplicates');

        /*  rows_to_remove = ONLY the extra copies deleted by keep='first'.
            duplicate_row_count is now also set to this same value in the
            backend, so both fields are safe fallbacks for each other.    */
        var toRemove = d.rows_to_remove !== undefined ? d.rows_to_remove
            : d.duplicate_row_count !== undefined ? d.duplicate_row_count : 0;
        var totalRows = d.total_rows || 0;
        var afterDrop = d.unique_rows !== undefined ? d.unique_rows : (totalRows - toRemove);

        /* ── Top stat tiles ── */
        var elC = document.getElementById('d-c');
        var elU = document.getElementById('d-u');
        var elP = document.getElementById('d-p');
        if (elC) elC.textContent = toRemove;
        if (elU) elU.textContent = afterDrop;
        if (elP) elP.textContent = totalRows > 0
            ? (toRemove / totalRows * 100).toFixed(1) + '%' : '0.0%';

        var prev = document.getElementById('d-prev');
        var btn = document.getElementById('dupBtn');

        /* ── CLEAN ── */
        if (toRemove === 0) {
            if (prev) prev.innerHTML =
                '<div style="padding:14px 16px;background:#f0fdf4;border:1.5px solid #bbf7d0;' +
                'border-radius:10px;display:flex;align-items:center;gap:12px">' +
                '<span style="font-size:22px">\u2705</span>' +
                '<div><div style="font-weight:700;color:#166534;font-size:13px">No duplicate rows found</div>' +
                '<div style="font-size:11px;color:#15803d;margin-top:2px">' +
                'Your dataset has no repeated rows \u2014 nothing to remove</div></div></div>';
            if (btn) btn.disabled = true;
            markDone(2);
            _refreshBadges();
            return;
        }

        /* ── BUILD HTML ── */
        var html = '';

        /* 1. Small informational line (no warning colors) */
        html +=
            '<div style="font-size:12px;color:#475569;margin-bottom:10px">' +
            '<b style="color:#be123c">' + toRemove + '</b> duplicate row' +
            (toRemove !== 1 ? 's' : '') + ' found and will be removed. ' +
            '<b style="color:#166534">' + afterDrop + '</b> row' +
            (afterDrop !== 1 ? 's' : '') + ' will remain in your dataset.' +
            '</div>';

        /* 2. Preview table: ONLY rows being deleted (first occurrence not shown) */
        if (d.preview && d.preview.rows && d.preview.rows.length) {
            var totalCount = d.preview.total_to_remove !== undefined
                ? d.preview.total_to_remove : toRemove;
            var isCapped = d.preview.preview_capped || (d.preview.rows.length < totalCount);
            var cols = d.preview.columns || [];

            var caption = isCapped
                ? 'Rows to be removed (showing ' + d.preview.rows.length + ' of ' + totalCount + ')'
                : 'Rows to be removed (' + d.preview.rows.length + ')';

            html +=
                '<div style="font-size:11px;font-weight:600;color:#64748b;margin-bottom:4px">' +
                caption + '</div>' +
                '<div style="overflow-x:auto;margin-bottom:10px">' +
                '<table class="dt"><thead><tr>' +
                '<th style="color:#94a3b8;font-size:10px;width:24px;text-align:center">#</th>' +
                cols.map(function (c) { return '<th>' + c + '</th>'; }).join('') +
                '</tr></thead><tbody>';

            html += d.preview.rows.map(function (row, i) {
                return '<tr><td style="color:#94a3b8;font-size:10px;text-align:center">' + (i + 1) + '</td>' +
                    row.map(function (cell) {
                        return '<td>' + (cell === null || cell === undefined
                            ? '<span style="color:#cbd5e1;font-style:italic">—</span>' : String(cell)) + '</td>';
                    }).join('') + '</tr>';
            }).join('');

            html += '</tbody></table></div>';
        }


        if (prev) prev.innerHTML = html;
        if (btn) btn.disabled = false;
        _refreshBadges();

    } catch (e) {
        var prev = document.getElementById('d-prev');
        if (prev) prev.innerHTML =
            '<div style="padding:10px;color:#ef4444;font-size:12px">' +
            'Failed to load duplicate info. Check backend connection.</div>';
    }
}


async function applyDup() {
    var r = document.querySelector('input[name="dKeep"]:checked');
    setBtnLoading('dupBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/duplicates', { keep: r ? r.value : 'first' });
        toast('Duplicates removed');
        await Promise.all([_refresh(), loadDups()]);
        markDone(2);
    } catch (e) { toast(e.message, 'terr'); }
    finally { setBtnLoading('dupBtn', false); }
}


/* ── CONSTANT COLUMNS ────────────────────────── */
async function loadConst() {
    try {
        var d = await req('GET', '/preprocessing/' + SID + '/constant-columns');
        var el = document.getElementById('constL'), btn = document.getElementById('constBtn');
        if (!d.constant_columns || !d.constant_columns.length) {
            el.innerHTML = '<div style="padding:10px;background:#f0fdf4;border-radius:7px;color:#166534;font-size:12px;font-weight:600">No constant columns found!</div>';
            btn.disabled = true; markDone(3);
        } else {
            el.innerHTML = d.constant_columns.map(function (c) {
                return '<label style="display:flex;align-items:center;gap:10px;padding:8px 10px;background:#fff7ed;border:1px solid #fed7aa;border-radius:7px;margin-bottom:5px;cursor:pointer;font-size:13px">'
                    + '<input type="checkbox" class="ccb" value="' + c.column + '" style="width:15px;height:15px;flex-shrink:0">'
                    + '<div><b>' + c.column + '</b> - always <code style="background:#fef3c7;padding:1px 4px;border-radius:3px;font-size:11px">' + c.constant_value + '</code></div></label>';
            }).join('');
            btn.disabled = false;
        }
    } catch (e) { /* silent */ }
}

async function applyConst() {
    var sel = Array.from(document.querySelectorAll('.ccb:checked')).map(function (c) { return c.value; });
    if (!sel.length) { toast('Select at least one column', 'terr'); return; }
    setBtnLoading('constBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/constant-columns', { columns: sel });
        toast(sel.length + ' columns removed');
        markDone(3);
        await Promise.all([_refresh(), loadConst()]);
    } catch (e) { toast(e.message, 'terr'); }
    finally { setBtnLoading('constBtn', false); }
}

/* ── ENCODING ────────────────────────────────── */
function eTab(name) {
    document.querySelectorAll('.ep').forEach(function (p) { p.style.display = 'none'; });
    document.getElementById('et-' + name).style.display = '';
    document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('on'); if (t.getAttribute('data-et') === name) t.classList.add('on'); });
}

function _markEncoded(cols) {
    cols.forEach(function (c) { encodedCols.add(c); });
    buildPicker('label', catCols, 'cat', encodedCols);
    buildPicker('onehot', catCols, 'cat', encodedCols);
    buildPicker('target', catCols, 'cat', encodedCols);
    var unenc = catCols.filter(function (c) { return !encodedCols.has(c); });
    popSel('ordC', unenc, 'Select column...');
}

async function applyLabel() {
    var avail = catCols.filter(function (c) { return !encodedCols.has(c); });
    var cols = getPicked('label'); if (!cols.length) cols = avail;
    if (!cols.length) { toast('No unencoded categorical columns left', 'terr'); return; }
    setBtnLoading('applyLabelBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/encoding/label', { columns: cols });
        toast('Label encoding applied on ' + cols.length + ' cols');
        _markEncoded(cols); markDone(4); _refresh();
    } catch (e) { toast(e.message, 'terr'); }
    finally { setBtnLoading('applyLabelBtn', false); }
}

async function applyOnehot() {
    var avail = catCols.filter(function (c) { return !encodedCols.has(c); });
    var cols = getPicked('onehot'); if (!cols.length) cols = avail;
    if (!cols.length) { toast('No unencoded categorical columns left', 'terr'); return; }
    var drop = document.getElementById('ohDrop').checked;
    setBtnLoading('applyOHBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/encoding/onehot', { columns: cols, drop_first: drop, handle_binary: true });
        toast('One-hot encoding applied');
        _markEncoded(cols); markDone(4);
        // After one-hot, column count changes — refresh cols and stats in parallel
        Promise.all([_refresh(), loadCols()]);
    } catch (e) { toast(e.message, 'terr'); }
    finally { setBtnLoading('applyOHBtn', false); }
}

async function applyOrdinal() {
    var col = document.getElementById('ordC').value;
    if (!col) { toast('Select a column', 'terr'); return; }
    var auto = document.getElementById('ordA').checked, cats = null;
    if (!auto) { var raw = (document.getElementById('ordO').value || '').trim(); if (!raw) { toast('Enter category order', 'terr'); return; } cats = raw.split(',').map(function (s) { return s.trim(); }).filter(Boolean); }
    setBtnLoading('applyOrdBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/encoding/ordinal', { column: col, auto_order: auto, categories: cats });
        toast('Ordinal encoding applied on ' + col);
        _markEncoded([col]); markDone(4); _refresh();
    } catch (e) { toast(e.message, 'terr'); }
    finally { setBtnLoading('applyOrdBtn', false); }
}

async function applyTarget() {
    var avail = catCols.filter(function (c) { return !encodedCols.has(c); });
    var cols = getPicked('target'); if (!cols.length) cols = avail;
    var tgt = document.getElementById('tgtC').value;
    if (!cols.length || !tgt) { toast('Select columns and a target column', 'terr'); return; }
    setBtnLoading('applyTgtBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/encoding/target', { columns: cols, target_column: tgt });
        toast('Target encoding applied');
        _markEncoded(cols); markDone(4); _refresh();
    } catch (e) { toast(e.message, 'terr'); }
    finally { setBtnLoading('applyTgtBtn', false); }
}

/* ── SCALING ─────────────────────────────────── */
function _updateScalingUI() {
    /* Columns that still haven't been scaled */
    var unscaled = numCols.filter(function (c) { return !scaledCols.has(c); });
    var ctrl = document.getElementById('scale-controls');
    var done = document.getElementById('scale-done');

    if (unscaled.length === 0 && numCols.length > 0) {
        /* Every column is scaled — show done panel */
        if (ctrl) ctrl.style.display = 'none';
        if (done) done.style.display = '';
    } else {
        /* Rebuild picker with only remaining unscaled columns */
        buildPicker('scale', unscaled.length > 0 ? unscaled : numCols, 'num');
        if (ctrl) ctrl.style.display = '';
        if (done) done.style.display = 'none';
    }
}

async function applyScaling() {
    /* Only offer unscaled columns; fall back to all if nothing selected */
    var unscaled = numCols.filter(function (c) { return !scaledCols.has(c); });
    var cols = getPicked('scale');
    if (!cols.length) cols = unscaled;
    if (!cols.length) { toast('No unscaled numerical columns left', 'terr'); return; }
    var m = (document.querySelector('input[name="scM"]:checked') || {}).value || 'standard';
    setBtnLoading('applyScaleBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/scaling', { columns: cols, method: m });
        toast(m + ' scaling applied on ' + cols.length + ' col' + (cols.length > 1 ? 's' : ''));
        markDone(5);
        /* Track scaled columns; clear redo stack (new action invalidates redo history) */
        cols.forEach(function (c) { scaledCols.add(c); });
        _scalingStack.push(cols.slice());
        _scalingRedoStack = [];
        _updateScalingUI();
        _refresh();
    } catch (e) { toast(e.message, 'terr'); }
    finally { setBtnLoading('applyScaleBtn', false); }
}

/* ── OUTLIERS ────────────────────────────────── */
async function applyOut() {
    var cols = getPicked('out'); if (!cols.length) cols = numCols;
    if (!cols.length) { toast('No numerical columns', 'terr'); return; }
    var det = (document.querySelector('input[name="oD"]:checked') || {}).value || 'iqr';
    var act = (document.querySelector('input[name="oA"]:checked') || {}).value;
    if (!act) { toast('Select an action', 'terr'); return; }
    setBtnLoading('applyOutBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/outliers/handle', { columns: cols, method: det, action: act, threshold: 3.0 });
        toast('Outliers ' + act + 'ped using ' + det.toUpperCase());
        markDone(6); _refresh();
    } catch (e) { toast(e.message, 'terr'); }
    finally { setBtnLoading('applyOutBtn', false); }
}

/* ── SAMPLING ────────────────────────────────── */
async function loadDist() {
    var tgt = document.getElementById('sampT').value, box = document.getElementById('distB');
    if (!tgt) { box.style.display = 'none'; return; }
    if (!_plotlyLoaded) {
        await new Promise(function (res, rej) { var s = document.createElement('script'); s.src = 'https://cdn.plot.ly/plotly-2.26.0.min.js'; s.onload = res; s.onerror = rej; document.head.appendChild(s); });
        _plotlyLoaded = true;
    }
    try {
        var d = await req('GET', '/preprocessing/' + SID + '/sampling/distribution?target_column=' + encodeURIComponent(tgt));
        box.style.display = '';
        var keys = Object.keys(d.distribution || []), vals = keys.map(function (k) { return d.distribution[k]; });
        Plotly.newPlot('distP', [{ x: keys, y: vals, type: 'bar', marker: { color: ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6'].slice(0, keys.length) } }],
            { height: 195, margin: { l: 36, r: 8, t: 6, b: 36 }, paper_bgcolor: '#fff', plot_bgcolor: '#f8fafc', font: { size: 11 }, showlegend: false }, { displayModeBar: false });
        var ratio = d.imbalance_ratio;
        document.getElementById('distM').textContent = ratio ? 'Ratio: ' + ratio.toFixed(2) + ':1. ' + (ratio > 3 ? 'Significant imbalance.' : 'Mild imbalance.') : '';
    } catch (e) { box.style.display = 'none'; }
}

async function applySamp() {
    var tgt = document.getElementById('sampT').value;
    if (!tgt) { toast('Select a target column', 'terr'); return; }
    var m = (document.querySelector('input[name="sM"]:checked') || {}).value || 'smote';
    setBtnLoading('applySampBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/sampling', { target_column: tgt, method: m });
        toast(m.toUpperCase() + ' sampling applied');
        markDone(7); _refresh();
    } catch (e) { toast(e.message, 'terr'); }
    finally { setBtnLoading('applySampBtn', false); }
}

/* ── UNDO / REDO / RESET ─────────────────────── */
async function undoA() {
    try {
        /* Peek at history BEFORE undo so we know what's being undone */
        var histBefore = await req('GET', '/preprocessing/' + SID + '/history').catch(function () { return []; });
        var lastEntry = histBefore.length > 0 ? histBefore[histBefore.length - 1] : null;
        var isScaling = lastEntry && lastEntry.description &&
            lastEntry.description.toLowerCase().indexOf('scal') !== -1;

        await req('POST', '/preprocessing/' + SID + '/undo');
        toast('Undone', 'tinf');

        if (isScaling && _scalingStack.length > 0) {
            var batch = _scalingStack.pop();
            _scalingRedoStack.push(batch);          // save for possible redo
            batch.forEach(function (c) { scaledCols.delete(c); });
        }
        _updateScalingUI();

        /* ── LIVE REFRESH: update stats + badge + currently visible section ── */
        await _fullRefresh();

    } catch (e) { toast('Nothing to undo', 'terr'); }
}

async function redoA() {
    try {
        /* Record history length before redo to detect what was added */
        var histBefore = await req('GET', '/preprocessing/' + SID + '/history').catch(function () { return []; });
        var szBefore = histBefore.length;

        await req('POST', '/preprocessing/' + SID + '/redo');
        toast('Redone', 'tinf');

        var histAfter = await req('GET', '/preprocessing/' + SID + '/history').catch(function () { return []; });
        if (histAfter.length > szBefore) {
            var redone = histAfter[histAfter.length - 1];
            if (redone && redone.description &&
                redone.description.toLowerCase().indexOf('scal') !== -1 &&
                _scalingRedoStack.length > 0) {
                var batch = _scalingRedoStack.pop();
                _scalingStack.push(batch);
                batch.forEach(function (c) { scaledCols.add(c); });
            }
        }
        _updateScalingUI();

        /* ── LIVE REFRESH: update stats + badge + currently visible section ── */
        await _fullRefresh();

    } catch (e) { toast('Nothing to redo', 'terr'); }
}

async function resetA() {
    if (!confirm('Reset to original dataset? All changes will be lost.')) return;
    try {
        await req('POST', '/preprocessing/' + SID + '/reset');
        toast('Reset done', 'tinf');
        encodedCols.clear();
        /* Clear all scaling tracking */
        scaledCols.clear();
        _scalingStack = [];
        _scalingRedoStack = [];
        _updateScalingUI();
        /* Remove all done markers */
        document.querySelectorAll('.nav-btn').forEach(function (b) { b.classList.remove('done'); });
        /* Reload columns and fully refresh everything live */
        await Promise.all([loadStats(true), loadCols()]);
        await _fullRefresh();
        loadHist();
    } catch (e) { toast(e.message, 'terr'); }
}

/* ── REPORT ──────────────────────────────────── */
async function loadReport() {
    try {
        var [s, hist, missing, dups] = await Promise.all([
            req('GET', '/preprocessing/' + SID + '/stats'),
            req('GET', '/preprocessing/' + SID + '/history').catch(function () { return []; }),
            req('GET', '/preprocessing/' + SID + '/missing-values').catch(function () { return null; }),
            req('GET', '/preprocessing/' + SID + '/duplicates').catch(function () { return null; })
        ]);
        var orig = origShape || { rows: s.current_rows, cols: s.current_columns };
        document.getElementById('r-or').textContent = orig.rows;
        document.getElementById('r-cr').textContent = s.current_rows;
        document.getElementById('r-oc').textContent = orig.cols;
        document.getElementById('r-cc').textContent = s.current_columns;
        document.getElementById('r-st').textContent = hist ? hist.length : 0;
        var se = document.getElementById('rSteps');
        if (!hist || !hist.length) { se.innerHTML = '<p style="font-size:12px;color:#94a3b8;font-style:italic">No steps applied yet.</p>'; }
        else {
            se.innerHTML = hist.map(function (h, i) {
                return '<div style="display:flex;align-items:center;gap:8px;padding:6px 8px;background:#f8fafc;border-radius:7px;margin-bottom:4px;font-size:12px">'
                    + '<span style="width:18px;height:18px;border-radius:50%;background:#6366f1;color:#fff;font-size:9px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0">' + (i + 1) + '</span>'
                    + '<span>' + h.description + '<span style="color:#94a3b8;font-size:10px;margin-left:6px">' + new Date(h.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + '</span></span></div>';
            }).join('');
        }
        var qr = [
            { c: 'Missing Values', ok: missing && missing.total_null_count === 0, d: missing ? missing.total_null_count + ' remaining (' + missing.total_null_percentage + '%)' : 'Unknown' },
            { c: 'Duplicate Rows', ok: dups && (dups.rows_to_remove || 0) === 0, d: dups ? (dups.rows_to_remove || 0) + ' rows to remove, ' + (dups.unique_rows || 0) + ' will remain' : 'Unknown' },
            { c: 'Rows', ok: true, d: orig.rows + ' to ' + s.current_rows },
            { c: 'Columns', ok: true, d: orig.cols + ' to ' + s.current_columns },
            { c: 'Steps Done', ok: hist && hist.length > 0, d: (hist ? hist.length : 0) + ' steps' }
        ];
        document.getElementById('rQual').innerHTML = qr.map(function (r) {
            return '<tr><td style="font-weight:600">' + r.c + '</td><td><span class="b ' + (r.ok ? 'bg' : 'by') + '">' + (r.ok ? 'OK' : 'Check') + '</span></td><td style="color:#64748b">' + r.d + '</td></tr>';
        }).join('');
    } catch (e) { toast('Report load failed', 'terr'); }
}

function exportCSV() {
    var a = document.createElement('a');
    a.href = API + '/preprocessing/' + SID + '/export-csv';
    a.download = 'preprocessed.csv';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    toast('Downloading...', 'tinf');
}
function goTrain() { window.location.href = 'training.html?session_id=' + SID; }
