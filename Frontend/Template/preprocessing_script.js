/* preprocessing_script.js - optimized for maximum speed */
var API = 'http://127.0.0.1:8000/api';
var SID = null;
var numCols = [], catCols = [], allCols = [];
var encodedCols = new Set();
var origShape = null;
var _plotlyLoaded = false;
var _pickers = {};

/* ── Info texts ─────────────────────────────── */
var INFO = {
    missing: { t: 'Missing Values', b: 'Missing values are empty cells. Most ML models fail when they encounter them.<br><br><b>Numerical:</b> Fill with Mean, Median, a custom number, or drop rows with missing data.<br><b>Categorical:</b> Fill with Mode (most common), a custom word, or drop the row.<br><br>If no columns are selected, the operation applies to every column of that type.' },
    duplicates: { t: 'Duplicate Rows', b: 'Duplicate rows are identical across every column. They add no new information and bias model training.<br><br>You can keep the first occurrence or the last one when removing.' },
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
    closeSB();
    if (name === 'missing') loadMissing();
    else if (name === 'duplicates') loadDups();
    else if (name === 'constant') loadConst();
    else if (name === 'report') loadReport();
}
function markDone(n) { var el = document.getElementById('n' + n); if (el) { el.textContent = 'ok'; el.parentElement.classList.add('done'); } }

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
    buildPicker('numMis', numCols, 'num');
    buildPicker('catMis', catCols, 'cat');
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

/* ── After-action refresh (stats + history in parallel) ── */
async function _refresh() {
    return Promise.all([
        req('GET', '/preprocessing/' + SID + '/stats').then(function (d) {
            if (!origShape) origShape = { rows: d.original_rows || d.current_rows, cols: d.original_columns || d.current_columns };
            document.getElementById('dsN').textContent = d.dataset_name || 'Dataset';
            document.getElementById('dsS').textContent = d.current_rows + ' rows, ' + d.current_columns + ' cols';
            document.getElementById('undoB').disabled = !d.can_undo;
            document.getElementById('redoB').disabled = !d.can_redo;
        }).catch(function () { }),
        loadHist()
    ]);
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
        if (!d.columns || !d.columns.length) { tb.innerHTML = '<tr><td colspan="5" style="padding:12px;text-align:center;color:#94a3b8">No columns</td></tr>'; return; }
        tb.innerHTML = d.columns.map(function (c) {
            var lv = c.null_percentage > 50 ? ['br', 'High'] : c.null_percentage > 20 ? ['by', 'Med'] : c.null_percentage > 0 ? ['bgr', 'Low'] : ['bg', 'None'];
            return '<tr><td><b>' + c.column + '</b></td><td>' + c.data_type + '</td><td>' + c.null_count + '</td><td>' + c.null_percentage + '%</td><td><span class="b ' + lv[0] + '">' + lv[1] + '</span></td></tr>';
        }).join('');
        if ((d.total_null_count || 0) === 0) markDone(1);
    } catch (e) { /* silent */ }
}

async function applyNM() {
    var cols = getPicked('numMis');
    var radio = document.querySelector('input[name="nMis"]:checked');
    if (!radio) { toast('Pick a fill method', 'terr'); return; }
    var s = radio.value, body = { columns: cols.length ? cols : null, strategy: s };
    if (s === 'constant_num') body.constant_value = parseFloat(document.getElementById('cNum').value) || 0;
    setBtnLoading('applyNMBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/missing-values', body);
        toast('Numerical columns updated');
        markDone(1);
        Promise.all([_refresh(), loadMissing()]);
    } catch (e) { toast(e.message, 'terr'); }
    finally { setBtnLoading('applyNMBtn', false); }
}

async function applyCM() {
    var cols = getPicked('catMis');
    var radio = document.querySelector('input[name="cMis"]:checked');
    if (!radio) { toast('Pick a fill method', 'terr'); return; }
    var s = radio.value, body = { columns: cols.length ? cols : null, strategy: s };
    if (s === 'constant_cat') body.constant_string = document.getElementById('cCat').value || 'Unknown';
    setBtnLoading('applyCMBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/missing-values', body);
        toast('Categorical columns updated');
        markDone(1);
        Promise.all([_refresh(), loadMissing()]);
    } catch (e) { toast(e.message, 'terr'); }
    finally { setBtnLoading('applyCMBtn', false); }
}

/* ── DUPLICATES ──────────────────────────────── */
async function loadDups() {
    try {
        var d = await req('GET', '/preprocessing/' + SID + '/duplicates');
        document.getElementById('d-c').textContent = d.duplicate_row_count || 0;
        document.getElementById('d-p').textContent = ((d.duplicate_row_count || 0) / Math.max(d.total_rows || 1, 1) * 100).toFixed(1) + '%';
        document.getElementById('d-u').textContent = (d.total_rows || 0) - (d.duplicate_row_count || 0);
        var prev = document.getElementById('d-prev');
        if (!d.duplicate_row_count) {
            prev.innerHTML = '<div style="padding:10px;background:#f0fdf4;border-radius:7px;color:#166534;font-size:12px;font-weight:600">No duplicates - your data is clean!</div>';
            document.getElementById('dupBtn').disabled = true; markDone(2);
        } else if (d.preview && d.preview.rows && d.preview.rows.length) {
            var c = d.preview.columns;
            var h = '<thead><tr>' + c.map(function (x) { return '<th>' + x + '</th>'; }).join('') + '</tr></thead>';
            var rows = d.preview.rows.slice(0, 5).map(function (row) { return '<tr>' + row.map(function (cell) { return '<td>' + cell + '</td>'; }).join('') + '</tr>'; }).join('');
            prev.innerHTML = '<div style="overflow-x:auto"><table class="dt">' + h + '<tbody>' + rows + '</tbody></table></div><div style="font-size:10px;color:#94a3b8;margin-top:3px">First 5 duplicate rows</div>';
            document.getElementById('dupBtn').disabled = false;
        } else {
            prev.innerHTML = '<div style="font-size:12px;color:#f59e0b;font-weight:600">' + d.duplicate_row_count + ' duplicates found</div>';
            document.getElementById('dupBtn').disabled = false;
        }
    } catch (e) { /* silent */ }
}

async function applyDup() {
    var r = document.querySelector('input[name="dKeep"]:checked');
    setBtnLoading('dupBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/duplicates', { keep: r ? r.value : 'first' });
        toast('Duplicates removed');
        markDone(2);
        Promise.all([_refresh(), loadDups()]);
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
        Promise.all([_refresh(), loadConst()]);
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
async function applyScaling() {
    var cols = getPicked('scale'); if (!cols.length) cols = numCols;
    if (!cols.length) { toast('No numerical columns', 'terr'); return; }
    var m = (document.querySelector('input[name="scM"]:checked') || {}).value || 'standard';
    setBtnLoading('applyScaleBtn', true);
    try {
        await req('POST', '/preprocessing/' + SID + '/scaling', { columns: cols, method: m });
        toast(m + ' scaling applied on ' + cols.length + ' cols');
        markDone(5); _refresh();
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
    try { await req('POST', '/preprocessing/' + SID + '/undo'); toast('Undone', 'tinf'); _refresh(); }
    catch (e) { toast('Nothing to undo', 'terr'); }
}
async function redoA() {
    try { await req('POST', '/preprocessing/' + SID + '/redo'); toast('Redone', 'tinf'); _refresh(); }
    catch (e) { toast('Nothing to redo', 'terr'); }
}
async function resetA() {
    if (!confirm('Reset to original dataset? All changes will be lost.')) return;
    try {
        await req('POST', '/preprocessing/' + SID + '/reset');
        toast('Reset done', 'tinf');
        encodedCols.clear();
        document.querySelectorAll('.nav-btn').forEach(function (b) { b.classList.remove('done'); });
        Promise.all([loadStats(true), loadCols()]).then(function () { loadMissing(); loadHist(); });
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
            { c: 'Duplicate Rows', ok: dups && dups.duplicate_row_count === 0, d: dups ? dups.duplicate_row_count + ' remaining' : 'Unknown' },
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
