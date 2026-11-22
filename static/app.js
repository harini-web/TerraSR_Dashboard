/* DeBlur Dashboard — all interactions are local to this application's server. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const defaults = {
    checkpoint: 'runs/exp1/best.pt', processor: 'Auto', mode: 'auto',
    bands: '1,2,3', offset: '', refl_max: '0.3',
    tile: '128', overlap: '16', max_output_mp: '8'
  };
  const state = {file: null, imageURL: null, latest: null, history: [], status: null};
  let busy = false;
  let timer = null;
  let started = 0;
  let settings = {...defaults};
  const accepted = new Set(['png', 'jpg', 'jpeg', 'tif', 'tiff', 'npz']);

  function svgIcon(name) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('icon');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', '#ico-' + name);
    svg.append(use); return svg;
  }

  function setView(name) {
    if (!$("view-" + name)) return;
    document.querySelectorAll('.view').forEach(el => el.classList.toggle('active', el.id === 'view-' + name));
    document.querySelectorAll('[data-view].nav-item').forEach(el => el.classList.toggle('active', el.dataset.view === name));
    const names = {dashboard: 'Dashboard', enhance: 'Enhance image', results: 'Recent results', guide: 'How to use', settings: 'Model settings'};
    $('breadcrumb-active').textContent = names[name] || name;
    $('sidebar').classList.remove('open'); $('scrim').classList.remove('visible');
    window.scrollTo({top: 0, behavior:'smooth'});
    if (name === 'results') fetchHistory();
    if (name === 'settings') populateSettingsForm();
  }
  document.querySelectorAll('[data-view]').forEach(btn => btn.addEventListener('click', () => setView(btn.dataset.view)));
  $('mobile-menu').addEventListener('click', () => { $('sidebar').classList.add('open'); $('scrim').classList.add('visible'); });
  $('scrim').addEventListener('click', () => { $('sidebar').classList.remove('open'); $('scrim').classList.remove('visible'); });

  function loadSettings() {
    try {
      const loaded = JSON.parse(localStorage.getItem('DeBlur_settings') || '{}');
      if (loaded && typeof loaded === 'object') {
        for (const key of Object.keys(defaults)) if (loaded[key] !== undefined) settings[key] = String(loaded[key]);
      }
    } catch (_) { /* private browser modes may disable storage */ }
  }
  function populateSettingsForm() {
    const form = $('settings-form');
    for (const name of Object.keys(defaults)) if (form.elements[name]) form.elements[name].value = settings[name];
  }
  function setSettingsFeedback(text, isError = false) {
    $('settings-feedback').textContent = text;
    $('settings-feedback').style.color = isError ? '#be5555' : '#247d61';
  }
  function validateSettings(s) {
    const bands = s.bands.split(',').map(v => Number(v.trim()));
    if (bands.length !== 3 || !bands.every(v => Number.isInteger(v) && v >= 1) || new Set(bands).size !== 3) return 'RGB bands must be different positive values such as 1,2,3.';
    const refl = Number(s.refl_max), tile = Number(s.tile), overlap = Number(s.overlap), maxMp = Number(s.max_output_mp);
    if (!(refl > 0 && refl <= 2)) return 'Reflectance stretch must be in the range (0, 2].';
    if (!Number.isInteger(overlap) || overlap < 0 || overlap > 32 || tile <= overlap * 2) return 'Tile overlap is invalid. Default is tile 128, overlap 16.';
    if (!Number.isInteger(maxMp) || maxMp < 1 || maxMp > 30) return 'Maximum output must be 1–30 megapixels.';
    if (s.offset.trim() !== '' && !Number.isFinite(Number(s.offset))) return 'Offset must be a number or left blank.';
    return '';
  }
  $('settings-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const next = {};
    for (const key of Object.keys(defaults)) next[key] = String(data.get(key) ?? '');
    const err = validateSettings(next);
    if (err) {setSettingsFeedback(err, true); return;}
    settings = next;
    try {localStorage.setItem('DeBlur_settings', JSON.stringify(settings));} catch (_) { /* optional */ }
    setSettingsFeedback('Settings saved. They will apply to your next image.');
    await fetchStatus();
  });
  $('reset-settings').addEventListener('click', async () => {
    settings = {...defaults}; populateSettingsForm();
    try {localStorage.removeItem('DeBlur_settings');} catch (_) { /* optional */ }
    setSettingsFeedback('Default settings restored.'); await fetchStatus();
  });

  function renderStatus(data) {
    state.status = data;
    const ready = data.checkpoint_present;
    $('dash-model').textContent = ready ? 'Ready' : 'Not found';
    $('enh-model').textContent = ready ? 'Ready' : 'Not found';
    $('enh-model-desc').textContent = ready ? 'Trained checkpoint located' : 'Add runs/exp1/best.pt';
    $('model-dot').classList.toggle('ready', ready);
    $('model-ready-title').textContent = ready ? 'Your model file is available' : 'Trained model file not found';
    $('model-ready-info').textContent = ready
      ? `${data.checkpoint_name} · ${data.gpu_available ? 'GPU detected' : 'CPU available'}`
      : 'Copy best.pt into runs/exp1, or change its path in Settings.';
    $('global-alert').classList.toggle('hidden', ready);
    $('global-alert-text').textContent = ready ? '' : 'Model not found. To run enhancement, add your trained best.pt or update its path in Model settings.';
    const count = Math.max(Number(data.current_count || 0), state.history.length);
    $('dash-count').textContent = String(count);
    $('enh-count').textContent = String(count);
    if (state.latest) $('dash-scale').textContent = state.latest.scale + '×';
  }
  async function fetchStatus() {
    try {
      const resp = await fetch('/api/status?checkpoint=' + encodeURIComponent(settings.checkpoint), {cache:'no-store'});
      if (!resp.ok) throw Error('Status unavailable');
      renderStatus(await resp.json());
    } catch (e) {
      $('model-ready-title').textContent = 'Unable to reach inference server';
      $('model-ready-info').textContent = 'Keep the terminal open and refresh this browser tab.';
      $('enh-model').textContent = 'Offline';
      $('dash-model').textContent = 'Offline';
      $('global-alert').classList.remove('hidden');
      $('global-alert-text').textContent = 'Cannot connect to the local Python server. Run python app.py in the project folder.';
    }
  }

  function revokePreview() {
    if (state.imageURL) URL.revokeObjectURL(state.imageURL);
    state.imageURL = null;
  }
  function setSelectedFile(file) {
    if (!file) return;
    const ext = (file.name.split('.').pop() || '').toLowerCase();
    if (!accepted.has(ext)) {message('Please choose PNG, JPG, TIFF or NPZ.', 'error'); return;}
    if (file.size > 80 * 1024 * 1024) {message('This file is larger than 80 MB. Please crop it first.', 'error'); return;}
    state.file = file;
    revokePreview();
    $('selected-file').classList.remove('hidden');
    $('filename').textContent = file.name;
    $('file-meta').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · Ready to enhance`;
    const preview = $('file-preview');
    preview.replaceChildren(svgIcon('file'));
    if (['png', 'jpg', 'jpeg'].includes(ext)) {
      state.imageURL = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => { $('file-meta').textContent = `${img.naturalWidth} × ${img.naturalHeight} px · ${(file.size / 1024 / 1024).toFixed(2)} MB`; };
      img.alt = ''; img.src = state.imageURL;
      preview.replaceChildren(img);
    }
    $('result-panel').classList.add('hidden');
    message('Image selected. Click the purple Enhance image now button.', 'success');
  }
  $('image-file').addEventListener('change', (e) => setSelectedFile(e.target.files[0]));
  const uploader = $('dropzone');
  const choose = () => $('image-file').click();
  uploader.addEventListener('click', choose);
  $('choose-file').addEventListener('click', e => {e.stopPropagation(); choose();});
  uploader.addEventListener('keydown', e => {if (e.key === 'Enter' || e.key === ' ') {e.preventDefault(); choose();}});
  ['dragenter', 'dragover'].forEach(type => uploader.addEventListener(type, e => {e.preventDefault(); uploader.classList.add('dragging');}));
  ['dragleave', 'drop'].forEach(type => uploader.addEventListener(type, e => {e.preventDefault(); uploader.classList.remove('dragging');}));
  uploader.addEventListener('drop', (e) => setSelectedFile(e.dataTransfer?.files?.[0]));
  $('remove-file').addEventListener('click', () => {state.file = null; $('image-file').value=''; $('selected-file').classList.add('hidden'); $('result-panel').classList.add('hidden'); revokePreview(); message('Image removed. Select an image to start.', 'success');});
  function message(text, type = 'working') {
    const box = $('process-message');
    box.classList.remove('hidden', 'error', 'success');
    if (type !== 'working') box.classList.add(type);
    $('process-message-text').textContent = text;
  }
  function processing(isBusy) {
    busy = isBusy;
    $('enhance-button').disabled = isBusy;
    $('enhance-button-text').textContent = isBusy ? 'Processing image...' : 'Enhance image now';
  }
  function humanSize(arr) {return `${Number(arr[0]).toLocaleString()} × ${Number(arr[1]).toLocaleString()} px`;}

  function renderResult(data) {
    state.latest = data;
    $('metric-input').textContent = humanSize(data.input_size);
    $('metric-output').textContent = humanSize(data.output_size);
    $('metric-scale').textContent = data.scale + '×';
    $('metric-time').textContent = Number(data.duration).toFixed(2) + ' s';
    $('dash-scale').textContent = data.scale + '×';
    $('preview-original').src = data.previews.input;
    $('preview-bicubic').src = data.previews.bicubic;
    $('preview-enhanced').src = data.previews.enhanced;
    $('download-native').href = data.download_url;
    $('download-preview').href = data.preview_download_url;
    $('download-native-label').textContent = data.geographic ? 'Download GeoTIFF' : 'Download enhanced PNG';
    $('download-hint').textContent = data.geographic
      ? 'The full-resolution GeoTIFF preserves geospatial coordinates. Download the PNG for a quick visual preview.'
      : 'Your enhanced image is a full-size PNG. You can also download a display-friendly preview.';
    const metrics = data.metrics;
    $('reference-metrics').classList.toggle('hidden', !metrics);
    if (metrics) {
      $('ref-psnr').textContent = metrics.psnr_cnn.toFixed(2) + ' dB';
      $('ref-ssim').textContent = metrics.ssim_cnn.toFixed(4);
      const delta = metrics.psnr_cnn - metrics.psnr_bicubic;
      $('ref-gain').textContent = (delta > 0 ? '+' : '') + delta.toFixed(2) + ' dB';
    }
    $('compare-after').src = data.previews.enhanced;
    $('compare-type').value = 'bicubic';
    $('compare-range').value = '50';
    updateComparison();
    $('result-panel').classList.remove('hidden');
    // Render the actual results as soon as the server responds.
    $('result-panel').scrollIntoView({behavior:'smooth', block:'start'});
  }
  function updateComparison() {
    if (!state.latest) return;
    const before = $('compare-type').value === 'input' ? state.latest.previews.input : state.latest.previews.bicubic;
    $('compare-before').src = before;
    $('compare-before').style.imageRendering = $('compare-type').value === 'input' ? 'pixelated' : 'auto';
    const value = Math.max(0, Math.min(100, Number($('compare-range').value)));
    $('compare-before').style.clipPath = `inset(0 ${100 - value}% 0 0)`;
    $('compare-divider').style.left = value + '%';
  }
  $('compare-range').addEventListener('input', updateComparison);
  $('compare-type').addEventListener('change', updateComparison);

  $('enhance-button').addEventListener('click', async () => {
    if (busy) return;
    if (!state.file) {message('First upload your low-resolution image using the box on the left.', 'error'); $('dropzone').focus(); return;}
    if (!state.status?.checkpoint_present) {message('Add your trained best.pt in runs/exp1 or select its path under Model settings.', 'error'); return;}
    const problem = validateSettings(settings);
    if (problem) {message(problem, 'error'); return;}
    processing(true);
    started = Date.now();
    message('Loading model and running CNN inference...');
    timer = setInterval(() => {
      $('process-message-text').textContent = `Processing your image on this computer... ${Math.floor((Date.now() - started)/1000)} s elapsed`;
    }, 1000);
    const data = new FormData();
    data.append('image', state.file);
    for (const [key, value] of Object.entries(settings)) data.append(key, value);
    try {
      const resp = await fetch('/api/enhance', {method: 'POST', body: data, cache:'no-store'});
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload.error || payload.detail || 'Unable to enhance image.');
      renderResult(payload);
      message(`Completed! ${humanSize(payload.input_size)} → ${humanSize(payload.output_size)} using CNN ${payload.scale}×.`, 'success');
      await fetchStatus();
      await fetchHistory();
    } catch (error) {
      message(error.message || 'Something went wrong. Check the Python terminal for details.', 'error');
    } finally {
      clearInterval(timer); processing(false);
    }
  });

  function createHistoryItem(item) {
    const el = document.createElement('article'); el.className = 'history-item';
    const icon = document.createElement('div'); icon.className = 'history-icon'; icon.append(svgIcon('image'));
    const copy = document.createElement('div'); copy.className = 'history-copy';
    const title = document.createElement('strong'); title.textContent = item.name;
    const small = document.createElement('small'); small.textContent = `${humanSize(item.input_size)} → ${humanSize(item.output_size)} · ${item.scale}× · ${item.duration.toFixed(2)} s`;
    copy.append(title, small);
    const actions = document.createElement('div'); actions.className = 'history-buttons';
    const open = document.createElement('button'); open.className = 'btn btn-outline'; open.textContent = 'View'; open.type = 'button';
    open.addEventListener('click', async () => {
      try {
        const resp = await fetch('/api/result/' + encodeURIComponent(item.id));
        if (!resp.ok) throw Error('This result has expired. Process the image again.');
        renderResult(await resp.json()); setView('enhance');
        setTimeout(() => $('result-panel').scrollIntoView({behavior: 'smooth'}), 100);
      } catch (error) {alert(error.message);}
    });
    const down = document.createElement('a'); down.className = 'btn btn-primary'; down.href = item.download_url; down.textContent = 'Download';
    actions.append(open, down); el.append(icon, copy, actions); return el;
  }
  async function fetchHistory() {
    try {
      const response = await fetch('/api/history');
      if (!response.ok) throw Error();
      const data = await response.json();
      state.history = data.items || [];
      const list = $('history-list');
      if (state.history.length) list.replaceChildren(...state.history.map(createHistoryItem));
      else {
        const wrapper = document.createElement('div'); wrapper.className = 'empty-state';
        wrapper.append(svgIcon('image'));
        const title = document.createElement('h3'); title.textContent = 'No results yet';
        const p = document.createElement('p'); p.textContent = 'Enhance an image to see it appear here.';
        const btn = document.createElement('button'); btn.className = 'btn btn-primary'; btn.textContent = 'Upload your first image'; btn.addEventListener('click', () => setView('enhance'));
        wrapper.append(title, p, btn); list.replaceChildren(wrapper);
      }
      $('dash-count').textContent = String(state.history.length);
      $('enh-count').textContent = String(state.history.length);
      if (state.history.length) $('dash-scale').textContent = state.history[0].scale + '×';
    } catch (e) { /* session-only history is optional */ }
  }
  loadSettings(); populateSettingsForm(); fetchStatus(); fetchHistory();
})();
