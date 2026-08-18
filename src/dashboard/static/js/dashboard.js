// src/dashboard/static/js/dashboard.js
// Vanilla-JS controller for the Haven-style caregiver dashboard. No
// framework -- fetch() against the Flask API in app.py and re-render the
// relevant section. Matches the pattern already used by this project
// before this rewrite (see the old index.html's inline script).

const STATUS_STYLE = {
  Normal:    { bg: '#EAF0DE', fg: '#405221', dot: '#6D8B3E' },
  Elevated:  { bg: '#F5EAB4', fg: '#7C6212', dot: '#A3811A' },
  Fighting:  { bg: '#F3D9C8', fg: '#7A2E12', dot: '#A6491C' },
  Fall:      { bg: '#F5D9C0', fg: '#8A3B12', dot: '#C15A1C' },
  Hazard:    { bg: '#E8C4C0', fg: '#7A1F1F', dot: '#A62E2E' },
  Emergency: { bg: '#3D1200', fg: '#FBEFE6', dot: '#FBEFE6' },
};
const PRIORITY_STYLE = {
  High:   { bg: '#F3D9C8', fg: '#7A2E12' },
  Medium: { bg: '#FBF6DE', fg: '#7C6212' },
  Low:    { bg: '#EAF0DE', fg: '#405221' },
};
const TYPE_STYLE = {
  'Violence Detected': { bg: '#F3D9C8', fg: '#7A2E12' },
  'Fall Detected':     { bg: '#F5D9C0', fg: '#8A3B12' },
  'Hazard Detected':   { bg: '#E8C4C0', fg: '#7A1F1F' },
  'Emergency':         { bg: '#3D1200', fg: '#FBEFE6' },
  'Clip Ready':        { bg: '#EAF0DE', fg: '#405221' },
};
// Event types that count as an "alert" for the banner, zone status, and
// the notification sound below -- kept in one place so all three agree.
const ALERT_TYPES = ['Violence Detected', 'Fall Detected', 'Hazard Detected', 'Emergency'];

const NAV_ITEMS = [
  { key: 'dashboard', label: 'Dashboard' },
  { key: 'persona', label: 'My profile' },
  { key: 'cameras', label: 'Cameras' },
  { key: 'incidents', label: 'Incident history' },
  { key: 'clips', label: 'Clip archive' },
  { key: 'alerts', label: 'Alerts & notifications' },
  { key: 'analytics', label: 'Analytics' },
  { key: 'system', label: 'System settings' },
];

const Haven = {
  state: {
    page: 'dashboard',
    cameFrom: 'incidents',
    isAdmin: false,
    cameras: [],
    events: [],
    clips: [],
    settings: { recipients: [], threshold: 90, cooldown: 120, email_channel: true, sound_channel: true, desktop_channel: true },
    systemSettings: {},
    selectedIncidentId: null,
    editingCamera: null,
    deleteConfirmId: null,
    emailPopupPending: null,
    openPreviews: new Set(), // camera ids currently streaming a live preview
    notifiedAlertIds: new Set(), // alert-event ids we've already played a sound for
    eventsLoadedOnce: false,     // first loadEvents() only establishes a baseline -- no sound
    audioCtx: null,              // created lazily on first user gesture (autoplay policy)
  },

  async init() {
    const root = document.getElementById('haven-root');
    this.state.isAdmin = root.dataset.isAdmin === '1';
    this.renderNav();
    this.startClock();
    this.renderCalendarStrip();
    // Browsers block audio until the page has seen a user gesture. Grab the
    // first click anywhere to unlock the AudioContext so a later alert sound
    // (fired from the 15s poll, with no gesture of its own) can actually play.
    document.addEventListener('click', () => this._unlockAudio(), { once: true });
    await Promise.all([this.loadCameras(), this.loadEvents(), this.loadSettings()]);
    this.loadAnnouncements();
    document.getElementById('dash-greet-name').textContent = (root.dataset.caregiverName || '').split(' ')[0];
    this.goPage('dashboard');
    setInterval(() => this.loadEvents(), 15000);
  },

  // ---------------- Alert sound ----------------
  _unlockAudio() {
    if (this.state.audioCtx) return;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    this.state.audioCtx = new Ctx();
  },
  playAlertSound() {
    if (this.state.settings.sound_channel === false) return;
    this._unlockAudio();
    const ctx = this.state.audioCtx;
    if (!ctx) return;
    if (ctx.state === 'suspended') ctx.resume();
    // Two-tone chime -- short and distinct from a UI click, not alarm-siren
    // harsh (caregivers may hear this often; it shouldn't be startling).
    const now = ctx.currentTime;
    [[880, now, 0.16], [1175, now + 0.16, 0.22]].forEach(([freq, start, dur]) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(0.22, start + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + dur);
      osc.connect(gain).connect(ctx.destination);
      osc.start(start);
      osc.stop(start + dur + 0.02);
    });
  },

  // ---------------- Desktop (OS) notification ----------------
  // Native OS toast via the Notification API. Only fires while this tab is
  // open somewhere (foreground or backgrounded) -- it can't reach a closed
  // browser or a locked phone; that would need Push API + a service worker
  // + a server-side sender, which is a separate, much larger feature.
  //
  // Browsers only allow Notification on a secure context (https, or
  // literally "localhost"). If this dashboard is reached over plain HTTP
  // via a LAN IP -- which `app.run(host="0.0.0.0", ...)` suggests it is --
  // requestPermission()/the constructor will silently fail. We detect that
  // and surface it in the UI rather than pretending the toggle works.
  desktopNotificationsSupported() {
    return ('Notification' in window) && window.isSecureContext;
  },
  async requestDesktopPermission() {
    if (!this.desktopNotificationsSupported()) return 'unsupported';
    if (Notification.permission === 'default') {
      try { return await Notification.requestPermission(); }
      catch (e) { return 'denied'; }
    }
    return Notification.permission; // 'granted' or 'denied'
  },
  showDesktopNotification(event) {
    if (this.state.settings.desktop_channel === false) return;
    if (!this.desktopNotificationsSupported()) return;
    if (Notification.permission !== 'granted') return;
    const pct = Math.round((event.confidence || 0) * 100);
    try {
      new Notification(`Haven: ${event.event_type}`, {
        body: `${esc(event.camera_id)} · ${esc(event.room)} · ${pct}% confidence`,
        tag: `haven-alert-${event.id}`, // de-dupes if the same alert fires the callback twice
      });
    } catch (e) {
      // Constructor throws on some insecure/embedded contexts even when
      // permission reads 'granted' -- fail quietly, the chime still fired.
    }
  },

  // ---------------- Navigation ----------------
  renderNav() {
    const nav = document.getElementById('nav-list');
    nav.innerHTML = NAV_ITEMS.map(item => `
      <button class="nav-btn" id="nav-${item.key}" onclick="Haven.goPage('${item.key}')">
        <span class="dot"></span>${item.label}
      </button>
    `).join('');
  },

  goPage(name) {
    this.state.page = name;
    document.querySelectorAll('.haven-page').forEach(el => el.style.display = 'none');
    const target = document.getElementById('page-' + name);
    if (target) target.style.display = '';
    NAV_ITEMS.forEach(item => {
      const btn = document.getElementById('nav-' + item.key);
      if (btn) btn.classList.toggle('active', item.key === name);
    });
    window.scrollTo(0, 0);

    if (name === 'persona') this.loadProfile();
    if (name === 'cameras') this.renderCameras();
    if (name === 'incidents') this.renderIncidents();
    if (name === 'clips') this.loadClips();
    if (name === 'alerts') this.renderAlertsPage();
    if (name === 'analytics') this.loadAnalytics();
    if (name === 'system') this.loadSystemSettings();
    if (name === 'dashboard') this.renderDashboard();
  },

  // ---------------- Clock + calendar ----------------
  startClock() {
    this.drawClock();
    setInterval(() => this.drawClock(), 30000);
  },

  drawClock() {
    const now = new Date();
    const svg = document.getElementById('haven-clock');
    const cx = 110, cy = 110, rTickIn = 88, rTickOut = 98, rNum = 72, rArc = 98;
    let ticks = '', numbers = '';
    for (let h = 0; h < 12; h++) {
      const a = (h / 12) * 2 * Math.PI - Math.PI / 2;
      const x1 = (cx + rTickIn * Math.cos(a)).toFixed(1), y1 = (cy + rTickIn * Math.sin(a)).toFixed(1);
      const x2 = (cx + rTickOut * Math.cos(a)).toFixed(1), y2 = (cy + rTickOut * Math.sin(a)).toFixed(1);
      ticks += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#D8CFAE" stroke-width="2"></line>`;
      const nx = (cx + rNum * Math.cos(a)).toFixed(1), ny = (cy + rNum * Math.sin(a) + 5).toFixed(1);
      const label = h === 0 ? 12 : h;
      numbers += `<text x="${nx}" y="${ny}" font-size="14" fill="#291100" text-anchor="middle" font-family="Figtree">${label}</text>`;
    }
    const hourA = ((now.getHours() % 12) + now.getMinutes() / 60) / 12 * 2 * Math.PI - Math.PI / 2;
    const minA = (now.getMinutes() / 60) * 2 * Math.PI - Math.PI / 2;
    const a8 = (8 / 12) * 2 * Math.PI - Math.PI / 2, a6 = (6 / 12) * 2 * Math.PI - Math.PI / 2;
    const arcPath = `M ${(cx + rArc * Math.cos(a8)).toFixed(1)} ${(cy + rArc * Math.sin(a8)).toFixed(1)} A ${rArc} ${rArc} 0 1 1 ${(cx + rArc * Math.cos(a6)).toFixed(1)} ${(cy + rArc * Math.sin(a6)).toFixed(1)}`;
    const hourX = (cx + 60 * Math.cos(hourA)).toFixed(1), hourY = (cy + 60 * Math.sin(hourA)).toFixed(1);
    const minX = (cx + 80 * Math.cos(minA)).toFixed(1), minY = (cy + 80 * Math.sin(minA)).toFixed(1);

    svg.innerHTML = `
      <circle cx="110" cy="110" r="98" fill="none" stroke="#EFE3D2" stroke-width="10"></circle>
      <path d="${arcPath}" fill="none" stroke="#C9A227" stroke-width="10" stroke-linecap="round"></path>
      ${ticks}${numbers}
      <line x1="110" y1="110" x2="${hourX}" y2="${hourY}" stroke="#291100" stroke-width="5" stroke-linecap="round"></line>
      <line x1="110" y1="110" x2="${minX}" y2="${minY}" stroke="#A6491C" stroke-width="3" stroke-linecap="round"></line>
      <circle cx="110" cy="110" r="6" fill="#291100"></circle>
    `;
    document.getElementById('clock-time-label').textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const end = new Date(now);
    end.setHours(6, 0, 0, 0);
    if (now.getHours() >= 6) end.setDate(end.getDate() + 1);
    const diffMs = Math.max(end - now, 0);
    const hrs = Math.floor(diffMs / 3600000), mins = Math.floor((diffMs % 3600000) / 60000);
    document.getElementById('shift-remaining-label').textContent = `${hrs}h ${mins}m left in shift`;
  },

  renderCalendarStrip() {
    const now = new Date();
    document.getElementById('cal-month').textContent = now.toLocaleDateString([], { month: 'long' });
    document.getElementById('cal-day').textContent = now.getDate();
    const week = document.getElementById('cal-week');
    const dayNames = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
    const start = new Date(now);
    start.setDate(now.getDate() - now.getDay());
    let html = '';
    for (let i = 0; i < 7; i++) {
      const d = new Date(start);
      d.setDate(start.getDate() + i);
      const isToday = d.toDateString() === now.toDateString();
      const bg = isToday ? 'rgba(255,255,255,0.92)' : 'transparent';
      const fg = isToday ? '#291100' : '#fff';
      html += `<div style="text-align:center;flex:1">
        <div style="font-size:10px;opacity:0.7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.04em">${dayNames[i]}</div>
        <div style="width:30px;height:30px;border-radius:999px;display:flex;align-items:center;justify-content:center;margin:0 auto;font-size:13px;font-weight:700;background:${bg};color:${fg}">${d.getDate()}</div>
      </div>`;
    }
    week.innerHTML = html;
  },

  // ---------------- Data loading ----------------
  async loadCameras() {
    const res = await fetch('/api/cameras');
    this.state.cameras = await res.json();
  },
  async loadEvents() {
    const res = await fetch('/api/events');
    this.state.events = await res.json();
    this._checkForNewAlerts();
    if (this.state.page === 'dashboard') this.renderDashboard();
    if (this.state.page === 'incidents') this.renderIncidents();
  },
  // Compares the freshly-fetched events against what we've already notified
  // on, and plays the alert sound once if anything genuinely new showed up.
  // The very first call (page load) only records a baseline -- otherwise
  // every pre-existing open incident would chime the moment the dashboard
  // opens.
  _checkForNewAlerts() {
    const openAlerts = this.state.events.filter(e =>
      ALERT_TYPES.includes(e.event_type) && !e.reviewed && !e.false_positive
    );
    const freshAlerts = this.state.eventsLoadedOnce
      ? openAlerts.filter(e => !this.state.notifiedAlertIds.has(e.id))
      : [];
    openAlerts.forEach(e => this.state.notifiedAlertIds.add(e.id));
    if (freshAlerts.length) {
      this.playAlertSound();
      // One OS toast per new alert, not one summary -- each names a
      // different room/camera and a caregiver needs to know which.
      freshAlerts.forEach(e => this.showDesktopNotification(e));
    }
    this.state.eventsLoadedOnce = true;
  },
  async loadClips() {
    const res = await fetch('/api/clips');
    this.state.clips = await res.json();
    this.renderClips();
  },
  async loadSettings() {
    const res = await fetch('/api/settings');
    this.state.settings = await res.json();
  },
  async loadAnnouncements() {
    const res = await fetch('/api/announcements');
    const items = await res.json();
    const list = document.getElementById('announcements-list');
    if (!list) return;
    list.innerHTML = items.slice().reverse().map(note => `
      <div style="padding:14px 0;border-bottom:1px solid rgba(43,26,8,0.08);display:flex;gap:14px;align-items:flex-start">
        ${note.icon ? `<img src="/static/img/${note.icon}" style="width:20px;height:20px;border-radius:999px;object-fit:cover;flex:none;margin-top:2px">`
                     : `<span style="width:8px;height:8px;border-radius:999px;background:#C9A227;flex:none;margin-top:6px"></span>`}
        <div style="flex:1;min-width:0">
          <div style="font-size:11px;color:rgba(43,26,8,0.5);margin-bottom:3px">${note.time}${note.author ? ' &middot; ' + esc(note.author) : ''}</div>
          <div style="font-size:13px;line-height:1.5">${esc(note.text)}</div>
        </div>
      </div>
    `).join('') || '<div style="padding:14px 0;font-size:13px;color:rgba(43,26,8,0.5)">No announcements yet.</div>';
  },
  async postAnnouncement() {
    const input = document.getElementById('new-announcement-text');
    const text = input.value.trim();
    if (!text) return;
    await fetch('/api/announcements', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
    input.value = '';
    this.loadAnnouncements();
  },

  // ---------------- Dashboard (home) ----------------
  zoneStatusFor(room) {
    const now = Date.now();
    const recent = this.state.events.filter(e => e.room === room && (now - new Date(e.timestamp).getTime()) < 15 * 60000);
    if (recent.some(e => e.event_type === 'Emergency')) return 'Emergency';
    if (recent.some(e => e.event_type === 'Fall Detected')) return 'Fall';
    if (recent.some(e => e.event_type === 'Hazard Detected')) return 'Hazard';
    if (recent.some(e => e.event_type === 'Violence Detected')) return 'Fighting';
    return 'Normal';
  },

  renderDashboard() {
    const cams = this.state.cameras.filter(c => c.active);
    const totalPatients = cams.reduce((sum, c) => sum + (c.patients || 0), 0);
    document.getElementById('total-patients').textContent = totalPatients;

    // Duty zones = one row per active camera's room.
    const zonesEl = document.getElementById('duty-zones-list');
    zonesEl.innerHTML = cams.map(c => {
      const status = this.zoneStatusFor(c.room);
      const style = STATUS_STYLE[status] || STATUS_STYLE.Normal;
      const pr = PRIORITY_STYLE[c.priority] || PRIORITY_STYLE.Medium;
      return `
      <div class="card" style="padding:18px 22px;display:flex;align-items:center;gap:16px">
        <div style="width:44px;height:44px;border-radius:999px;background:${style.bg};display:flex;align-items:center;justify-content:center;flex:none">
          <span style="width:12px;height:12px;border-radius:999px;background:${style.fg}"></span>
        </div>
        <div style="flex:1">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
            <span style="font-size:11px;letter-spacing:0.06em;text-transform:uppercase;background:${pr.bg};color:${pr.fg};padding:3px 10px;border-radius:999px">${c.priority || 'Medium'} priority</span>
            <span style="font-size:15px;font-weight:600">${esc(c.room)}</span>
          </div>
          <div style="font-size:13px;color:rgba(43,26,8,0.6)">${c.patients || 0} patients &middot; ${esc(c.id)}</div>
        </div>
        <span style="font-size:12px;background:${style.bg};color:${style.fg};padding:6px 14px;border-radius:999px;display:flex;align-items:center;gap:6px;flex:none"><span style="width:7px;height:7px;border-radius:999px;background:currentColor"></span>${status}</span>
      </div>`;
    }).join('') || '<div class="card" style="padding:18px;font-size:13px;color:rgba(43,26,8,0.5)">No cameras configured yet. Add one from the Cameras page.</div>';

    // Zone status bar chart (share of zones in each state).
    const counts = { Normal: 0, Fighting: 0, Fall: 0, Hazard: 0, Emergency: 0 };
    cams.forEach(c => { const s = this.zoneStatusFor(c.room); counts[s] = (counts[s] || 0) + 1; });
    const total = cams.length || 1;
    const bars = document.getElementById('zone-status-bars');
    bars.innerHTML = Object.entries(counts).map(([label, count]) => {
      const pct = Math.round((count / total) * 100);
      const color = (STATUS_STYLE[label] || STATUS_STYLE.Normal).dot;
      return `<div style="flex:1;display:flex;flex-direction:column;align-items:center;height:100%;justify-content:flex-end">
        <div style="width:70%;background:${color};height:${pct}%;min-height:2px"></div>
        <span style="font-size:10px;color:rgba(43,26,8,0.6);margin-top:8px">${label}</span>
      </div>`;
    }).join('');

    // Quick action tags
    const openIncidents = this.state.events.filter(e => !e.reviewed && !e.false_positive).length;
    const offlineCams = this.state.cameras.filter(c => c.active && c.liveStatus === 'Offline').length;
    document.getElementById('qa-open-incidents-tag').textContent = `${openIncidents} open`;
    document.getElementById('qa-offline-cams-tag').textContent = `${offlineCams} offline`;

    // Active alert banner: an unresolved Violence/Fall/Hazard/Emergency event in the last 10 minutes.
    const now = Date.now();
    const active = this.state.events.filter(e =>
      ALERT_TYPES.includes(e.event_type) &&
      !e.reviewed && !e.false_positive &&
      (now - new Date(e.timestamp).getTime()) < 10 * 60000
    ).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];

    const banner = document.getElementById('active-alert-banner');
    if (active) {
      banner.innerHTML = `
      <div style="background:#EFE3D2;border:1px solid rgba(122,46,18,0.25);border-radius:20px;padding:24px 28px;margin-bottom:28px;position:relative">
        <img src="/static/img/icon-alert-elsewhere.jpg" style="width:96px;height:96px;object-fit:contain;position:absolute;left:-10px;top:-14px;transform:rotate(-7deg);border-radius:12px">
        <div style="margin-left:112px;font-size:13px;letter-spacing:0.06em;text-transform:uppercase;color:#7A2E12;font-weight:700;white-space:nowrap;margin-bottom:14px">Active alert &middot; needs attention</div>
        <div style="margin-left:112px;font-size:16px;color:#5C2409;margin-bottom:18px;max-width:60ch">${esc(active.event_type)} on ${esc(active.camera_id)} (${esc(active.room)}) at ${Math.round((active.confidence || 0) * 100)}% confidence, ${esc(active.timestamp)}.</div>
        <div style="margin-left:112px;display:flex;gap:10px">
          <button class="pill-btn" style="background:#7A2E12;color:#F7F5E9" onclick="Haven.sendIncidentAlert('${active.id}')">Send alert email</button>
          <button class="pill-btn" style="background:#fff;border:1px solid rgba(122,46,18,0.35);color:#7A2E12" onclick="Haven.selectIncident('${active.id}','dashboard')">View incident</button>
        </div>
      </div>`;
    } else {
      banner.innerHTML = '';
    }
  },

  async sendIncidentAlert(incidentId) {
    const res = await fetch('/api/settings/test-alert', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) { alert(data.error || 'Could not send alert.'); return; }
    document.getElementById('email-popup-context').textContent = `Alert email sent for incident ${incidentId}.`;
    document.getElementById('email-popup-recipients').textContent = 'To: ' + data.recipients.join(', ');
    document.getElementById('email-popup-backdrop').style.display = 'flex';
  },

  // ---------------- Profile ----------------
  async loadProfile() {
    const res = await fetch('/api/profile');
    const p = await res.json();
    document.getElementById('profile-name').textContent = p.name;
    document.getElementById('profile-role-line').textContent = p.role === 'admin' ? 'Administrator' : 'Caregiver';
    document.getElementById('profile-about').value = p.about || '';
    document.getElementById('profile-notes').value = p.notes || '';
  },
  async saveProfile() {
    const about = document.getElementById('profile-about').value;
    const notes = document.getElementById('profile-notes').value;
    await fetch('/api/profile', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ about, notes }) });
    const toast = document.getElementById('profile-save-toast');
    toast.style.display = 'inline';
    setTimeout(() => toast.style.display = 'none', 2000);
  },

  // ---------------- Cameras ----------------
  renderCameras() {
    document.getElementById('add-camera-btn').style.display = this.state.isAdmin ? 'inline-block' : 'none';
    document.getElementById('camera-admin-note').style.display = this.state.isAdmin ? 'none' : 'block';

    const el = document.getElementById('cameras-list');
    el.innerHTML = this.state.cameras.map(cam => {
      const isLive = cam.liveStatus === 'Live';
      const bg = isLive ? '#EAF0DE' : '#F3D9C8', fg = isLive ? '#405221' : '#7A2E12';
      const previewOpen = this.state.openPreviews.has(cam.id);
      const editDeleteButtons = this.state.isAdmin ? `
          <button class="pill-btn pill-btn-outline" style="padding:8px 16px;font-size:12px" onclick="Haven.editCamera('${cam.id}')">Edit</button>
          <button class="pill-btn" style="padding:8px 16px;font-size:12px;background:none;border:1px solid #A6491C;color:#A6491C" onclick="Haven.requestDeleteCamera('${cam.id}')">Delete</button>` : '';
      return `
      <div class="card" style="padding:16px 22px">
        <div style="display:flex;align-items:center;gap:16px">
          <div style="width:44px;height:44px;border-radius:999px;background:${bg};display:flex;align-items:center;justify-content:center;flex:none">
            <span style="width:12px;height:12px;border-radius:999px;background:${fg}"></span>
          </div>
          <div style="flex:1;min-width:0">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
              <span style="font-size:15px;font-weight:600">${esc(cam.id)}</span>
              <span style="font-size:13px;color:rgba(43,26,8,0.6)">${esc(cam.room)}</span>
            </div>
            <div style="font-size:12px;font-family:monospace;color:rgba(43,26,8,0.5)">${esc(cam.source)} &middot; threshold ${cam.threshold}</div>
          </div>
          <span style="font-size:12px;background:${bg};color:${fg};padding:6px 14px;border-radius:999px;flex:none">${cam.liveStatus}</span>
          <div style="display:flex;gap:6px;flex:none">
            <button class="pill-btn pill-btn-outline" style="padding:8px 16px;font-size:12px" ${isLive ? '' : 'disabled title="Camera is offline, no feed to preview"'} onclick="Haven.togglePreview('${cam.id}')">${previewOpen ? 'Hide preview' : 'Preview'}</button>${editDeleteButtons}
          </div>
        </div>
        ${previewOpen ? `
        <div style="margin-top:14px;border-radius:16px;overflow:hidden;background:#111;line-height:0">
          <img src="/video_feed/${encodeURIComponent(cam.id)}" alt="Live feed for ${esc(cam.id)}" style="width:100%;max-height:360px;object-fit:contain;display:block" onerror="this.parentElement.innerHTML='<div style=&quot;padding:24px;text-align:center;color:#F3D9C8;font-size:13px;font-family:Figtree,sans-serif&quot;>Feed unavailable. The camera may have gone offline.</div>'">
        </div>` : ''}
      </div>`;
    }).join('') || '<div class="card" style="padding:18px;font-size:13px;color:rgba(43,26,8,0.5)">No cameras yet.</div>';
  },

  togglePreview(camId) {
    // Toggling closed removes the <img> from the DOM on next render, which
    // is what actually drops the MJPEG connection -- we don't want more
    // than the caregiver's open cards streaming video at once.
    if (this.state.openPreviews.has(camId)) {
      this.state.openPreviews.delete(camId);
    } else {
      this.state.openPreviews.add(camId);
    }
    this.renderCameras();
  },

  openAddCamera() {
    this.state.editingCamera = { id: `CAM-${String(this.state.cameras.length + 1).padStart(2, '0')}`, room: '', source: '', threshold: 0.9, active: true, patients: 0, priority: 'Medium', isNew: true };
    this._openCameraModal('Add camera');
  },
  editCamera(id) {
    const cam = this.state.cameras.find(c => c.id === id);
    this.state.editingCamera = { ...cam, isNew: false };
    this._openCameraModal('Edit camera');
  },
  _openCameraModal(title) {
    const ec = this.state.editingCamera;
    document.getElementById('camera-modal-title').textContent = title;
    document.getElementById('cam-id').value = ec.id;
    document.getElementById('cam-room').value = ec.room;
    document.getElementById('cam-source').value = ec.source;
    document.getElementById('cam-threshold').value = ec.threshold;
    document.getElementById('cam-threshold-label').textContent = ec.threshold;
    document.getElementById('cam-patients').value = ec.patients || 0;
    document.getElementById('cam-priority').value = ec.priority || 'Medium';
    document.getElementById('cam-active').checked = !!ec.active;
    document.getElementById('camera-modal-backdrop').style.display = 'flex';
  },
  closeCameraModal() {
    document.getElementById('camera-modal-backdrop').style.display = 'none';
    this.state.editingCamera = null;
  },
  async saveCamera() {
    const ec = this.state.editingCamera;
    const payload = {
      id: document.getElementById('cam-id').value.trim(),
      room: document.getElementById('cam-room').value.trim(),
      source: document.getElementById('cam-source').value.trim(),
      threshold: parseFloat(document.getElementById('cam-threshold').value),
      patients: parseInt(document.getElementById('cam-patients').value || '0', 10),
      priority: document.getElementById('cam-priority').value,
      active: document.getElementById('cam-active').checked,
    };
    if (!payload.id || !payload.room || !payload.source) { alert('Fill in camera ID, room, and source.'); return; }

    if (ec.isNew) {
      await fetch('/api/cameras/add', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    } else {
      await fetch(`/api/cameras/update/${encodeURIComponent(ec.id)}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    }
    this.closeCameraModal();
    await this.loadCameras();
    this.renderCameras();
  },
  requestDeleteCamera(id) {
    this.state.deleteConfirmId = id;
    document.getElementById('delete-modal-text').textContent = `${id} will stop being monitored immediately.`;
    document.getElementById('delete-modal-backdrop').style.display = 'flex';
  },
  cancelDelete() {
    this.state.deleteConfirmId = null;
    document.getElementById('delete-modal-backdrop').style.display = 'none';
  },
  async confirmDeleteCamera() {
    await fetch(`/api/cameras/delete/${encodeURIComponent(this.state.deleteConfirmId)}`, { method: 'DELETE' });
    this.cancelDelete();
    await this.loadCameras();
    this.renderCameras();
  },

  // ---------------- Incident history ----------------
  populateCameraFilterOptions() {
    const opts = ['<option value="all">All cameras</option>']
      .concat(this.state.cameras.map(c => `<option value="${esc(c.id)}">${esc(c.id)} &middot; ${esc(c.room)}</option>`));
    ['filter-camera', 'clip-filter-camera'].forEach(id => {
      const el = document.getElementById(id);
      if (el && el.options.length <= 1) el.innerHTML = opts.join('');
    });
  },

  renderIncidents() {
    this.populateCameraFilterOptions();
    const camF = document.getElementById('filter-camera').value;
    const typeF = document.getElementById('filter-type').value;
    const search = document.getElementById('filter-search').value.toLowerCase();

    let list = this.state.events.slice().reverse();
    if (camF && camF !== 'all') list = list.filter(e => e.camera_id === camF);
    if (typeF && typeF !== 'all') list = list.filter(e => e.event_type === typeF);
    if (search) list = list.filter(e => (e.room || '').toLowerCase().includes(search) || (e.camera_id || '').toLowerCase().includes(search));

    const el = document.getElementById('incidents-list');
    el.innerHTML = list.map(inc => {
      const style = TYPE_STYLE[inc.event_type] || { bg: '#EAF0DE', fg: '#405221' };
      const pct = Math.round((inc.confidence || 0) * 100);
      return `
      <div class="card" style="padding:16px 22px;display:flex;align-items:center;gap:16px;cursor:pointer" onclick="Haven.selectIncident('${inc.id}','incidents')">
        <div style="width:44px;height:44px;border-radius:999px;background:${style.bg};display:flex;align-items:center;justify-content:center;flex:none">
          <span style="width:12px;height:12px;border-radius:999px;background:${style.fg}"></span>
        </div>
        <div style="flex:1;min-width:0">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <span style="font-size:15px;font-weight:600">${esc(inc.camera_id)}</span>
            <span style="font-size:13px;color:rgba(43,26,8,0.6)">${esc(inc.room)}</span>
          </div>
          <div style="font-size:12px;color:rgba(43,26,8,0.5)">${esc(inc.timestamp)} &middot; ${inc.clip_path ? 'Clip available' : 'No clip'}${inc.reviewed ? ' &middot; Reviewed' : ''}</div>
        </div>
        <span style="font-size:12px;background:${style.bg};color:${style.fg};padding:6px 14px;border-radius:999px;flex:none">${esc(inc.event_type)}</span>
        <span class="display-font" style="font-size:16px;color:#A3811A;flex:none;width:56px;text-align:right">${pct}%</span>
      </div>`;
    }).join('');
    document.getElementById('incidents-empty').style.display = list.length ? 'none' : 'block';
  },

  selectIncident(id, from) {
    this.state.selectedIncidentId = id;
    this.state.cameFrom = from;
    this.goPage('incidentDetail');
    this.renderIncidentDetail();
  },
  backFromDetail() {
    this.goPage(this.state.cameFrom);
  },

  async renderIncidentDetail() {
    const res = await fetch(`/api/incidents/${encodeURIComponent(this.state.selectedIncidentId)}`);
    if (!res.ok) { document.getElementById('incident-detail-body').innerHTML = '<p>Incident not found.</p>'; return; }
    const inc = await res.json();
    const pct = Math.round((inc.confidence || 0) * 100);
    const states = inc.states || [];

    document.getElementById('incident-detail-body').innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:24px">
        <div>
          <h1 class="display-font" style="font-weight:500;font-size:24px;margin:0 0 6px">${esc(inc.id)}</h1>
          <p style="font-size:13px;color:rgba(43,26,8,0.6);margin:0">${esc(inc.timestamp)} &middot; ${esc(inc.camera_id)} &middot; ${esc(inc.room)}</p>
        </div>
        <div style="text-align:right">
          <div style="font-size:11px;color:rgba(43,26,8,0.5);text-transform:uppercase;letter-spacing:0.06em">Confidence</div>
          <div class="display-font" style="font-size:32px;color:#A3811A">${pct}%</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1.3fr 1fr;gap:24px">
        <div>
          <div class="clip-placeholder" style="margin-bottom:20px">
            ${inc.clip_path ? `<a href="/clips/${esc(inc.clip_path.split('/').pop())}" target="_blank" style="font-size:13px;color:#A3811A">&#9654; Open clip</a>`
                             : `<span style="font-family:monospace;font-size:12px;color:rgba(43,26,8,0.5)">No clip recorded for this incident</span>`}
          </div>
          <h4 style="font-size:14px;margin:0 0 14px">People tracked at alert time</h4>
          <div style="display:flex;flex-direction:column;gap:12px;margin-bottom:24px">
            ${states.length ? states.map(s => `
              <div class="card" style="padding:16px">
                <div style="font-weight:600;font-size:14px;margin-bottom:4px">Person ${esc(String(s.track_id !== undefined ? s.track_id : ''))}</div>
                <div style="font-size:13px;color:rgba(43,26,8,0.65)">State: ${esc(s.state || 'unknown')}${s.score !== undefined ? ' &middot; score ' + Math.round(s.score * 100) + '%' : ''}${s.fall_status && s.fall_status !== 'None' ? ' &middot; fall: ' + esc(s.fall_status) : ''}</div>
              </div>`).join('') : (inc.event_type === 'Hazard Detected'
                ? `<div class="disabled-note">Hazard events aren't tied to a specific tracked person -- see the rule detail in the Summary panel instead.</div>`
                : `<div class="disabled-note">No per-person state snapshot was recorded for this incident (older event, or the detector didn't have a live tracker at alert time).</div>`)}
          </div>
          <h4 style="font-size:14px;margin:0 0 10px">Notes</h4>
          <textarea id="incident-notes" class="pill-input" style="width:100%;min-height:80px;border-radius:16px" placeholder="Add a follow-up note for the day shift...">${esc(inc.notes || '')}</textarea>
          <div style="display:flex;gap:10px;margin-top:12px">
            <button class="pill-btn" style="background:#EAF0DE;color:#405221" onclick="Haven.saveIncidentNotes()">Save note</button>
            <button class="pill-btn" style="background:${inc.reviewed ? '#405221' : '#EAF0DE'};color:${inc.reviewed ? '#EAF0DE' : '#405221'}" onclick="Haven.markReviewed()">${inc.reviewed ? 'Reviewed ✓' : 'Mark reviewed'}</button>
            <button class="pill-btn pill-btn-outline" onclick="Haven.markFalsePositive()">${inc.false_positive ? 'Unmark false positive' : 'Mark false positive'}</button>
          </div>
        </div>
        <div>
          <h4 style="font-size:14px;margin:0 0 14px">Summary</h4>
          <div class="card" style="padding:16px;font-size:13px;line-height:1.7;color:rgba(43,26,8,0.75)">
            Event type: <strong>${esc(inc.event_type)}</strong><br>
            Confidence: <strong>${pct}%</strong><br>
            ${inc.detail ? `Detail: <strong>${esc(inc.detail)}</strong><br>` : ''}
            Reviewed: <strong>${inc.reviewed ? 'Yes' : 'No'}</strong><br>
            Marked false positive: <strong>${inc.false_positive ? 'Yes' : 'No'}</strong>
          </div>
        </div>
      </div>`;
  },
  async saveIncidentNotes() {
    const notes = document.getElementById('incident-notes').value;
    await fetch(`/api/incidents/${encodeURIComponent(this.state.selectedIncidentId)}/notes`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ notes }) });
    await this.loadEvents();
  },
  async markReviewed() {
    await fetch(`/api/incidents/${encodeURIComponent(this.state.selectedIncidentId)}/review`, { method: 'POST' });
    await this.loadEvents();
    this.renderIncidentDetail();
  },
  async markFalsePositive() {
    await fetch(`/api/incidents/${encodeURIComponent(this.state.selectedIncidentId)}/false-positive`, { method: 'POST' });
    await this.loadEvents();
    this.renderIncidentDetail();
  },

  // ---------------- Clip archive ----------------
  renderClips() {
    this.populateCameraFilterOptions();
    const camF = document.getElementById('clip-filter-camera').value;
    const sort = document.getElementById('clip-sort').value;
    let list = this.state.clips.slice();
    if (camF && camF !== 'all') list = list.filter(c => c.camera_id === camF);
    list.sort((a, b) => sort === 'oldest' ? a.mtime - b.mtime : b.mtime - a.mtime);

    const grid = document.getElementById('clips-grid');
    grid.innerHTML = list.map(c => `
      <div style="cursor:pointer" onclick="window.open('/clips/${encodeURIComponent(c.filename)}','_blank')">
        <div class="clip-thumb"><span style="font-family:monospace;font-size:11px;color:rgba(43,26,8,0.45)">&#9654; ${esc(c.filename)}</span></div>
        <div style="font-size:13px;font-weight:600">${esc(c.filename)}</div>
        <div style="font-size:12px;color:rgba(43,26,8,0.55)">${esc(c.camera_id)} &middot; ${esc(c.timestamp)}</div>
      </div>`).join('');
    document.getElementById('clips-empty').style.display = list.length ? 'none' : 'block';
  },

  // ---------------- Alerts & notifications ----------------
  renderAlertsPage() {
    const s = this.state.settings;
    document.getElementById('recipients-list').innerHTML = (s.recipients || []).map((email, i) => `
      <div style="display:flex;justify-content:space-between;align-items:center;background:#F0ECDA;border-radius:999px;padding:8px 8px 8px 16px">
        <span style="font-size:13px">${esc(email)}</span>
        <button style="background:#fff;border:none;border-radius:999px;width:26px;height:26px;cursor:pointer;color:#A6491C;font-size:14px" onclick="Haven.removeRecipient(${i})">&times;</button>
      </div>`).join('') || '<div style="font-size:13px;color:rgba(43,26,8,0.5)">No recipients yet.</div>';

    document.getElementById('threshold-range').value = s.threshold;
    document.getElementById('threshold-val').textContent = s.threshold + '%';
    document.getElementById('cooldown-range').value = s.cooldown;
    document.getElementById('cooldown-val').textContent = s.cooldown + 's';

    const toggle = document.getElementById('email-toggle');
    const knob = document.getElementById('email-toggle-knob');
    toggle.style.background = s.email_channel ? '#405221' : '#E1DBBE';
    knob.style.left = s.email_channel ? '23px' : '3px';

    const soundToggle = document.getElementById('sound-toggle');
    const soundKnob = document.getElementById('sound-toggle-knob');
    if (soundToggle && soundKnob) {
      const soundOn = s.sound_channel !== false;
      soundToggle.style.background = soundOn ? '#405221' : '#E1DBBE';
      soundKnob.style.left = soundOn ? '23px' : '3px';
    }

    const desktopToggle = document.getElementById('desktop-toggle');
    const desktopKnob = document.getElementById('desktop-toggle-knob');
    if (desktopToggle && desktopKnob) {
      const desktopOn = s.desktop_channel !== false;
      desktopToggle.style.background = desktopOn ? '#405221' : '#E1DBBE';
      desktopKnob.style.left = desktopOn ? '23px' : '3px';
    }
    const warning = document.getElementById('desktop-notif-warning');
    if (warning) {
      // Only show the "won't actually work" note when it's genuinely true
      // for this browser/origin -- not a blanket disclaimer.
      warning.style.display = this.desktopNotificationsSupported() ? 'none' : 'block';
    }
  },
  onThresholdChange() { document.getElementById('threshold-val').textContent = document.getElementById('threshold-range').value + '%'; },
  onCooldownChange() { document.getElementById('cooldown-val').textContent = document.getElementById('cooldown-range').value + 's'; },
  addRecipient() {
    const input = document.getElementById('new-recipient-email');
    const email = input.value.trim();
    if (!email || !email.includes('@')) return;
    this.state.settings.recipients = (this.state.settings.recipients || []).concat(email);
    input.value = '';
    this.renderAlertsPage();
  },
  removeRecipient(i) {
    this.state.settings.recipients.splice(i, 1);
    this.renderAlertsPage();
  },
  toggleEmailChannel() {
    this.state.settings.email_channel = !this.state.settings.email_channel;
    this.renderAlertsPage();
  },
  toggleSoundChannel() {
    this.state.settings.sound_channel = !(this.state.settings.sound_channel !== false);
    this.renderAlertsPage();
    // Unlock + preview immediately so the toggle click itself is the user
    // gesture that lets audio play (no waiting for the next real alert).
    if (this.state.settings.sound_channel) this.playAlertSound();
  },
  async toggleDesktopChannel() {
    const turningOn = !(this.state.settings.desktop_channel !== false);
    this.state.settings.desktop_channel = turningOn;
    if (turningOn) {
      // The permission prompt has to originate from this click (a user
      // gesture) -- it can't be requested later from the background poll.
      const result = await this.requestDesktopPermission();
      if (result === 'granted') {
        try { new Notification('Haven notifications enabled', { body: 'You\'ll get a desktop alert here when something needs attention.' }); }
        catch (e) { /* non-fatal preview */ }
      } else if (result === 'denied') {
        alert('Desktop notifications are blocked for this site in your browser settings. The sound alert will still play.');
      }
    }
    this.renderAlertsPage();
  },
  async saveAlertSettings() {
    const payload = {
      recipients: this.state.settings.recipients,
      threshold: parseInt(document.getElementById('threshold-range').value, 10),
      cooldown: parseInt(document.getElementById('cooldown-range').value, 10),
      email_channel: this.state.settings.email_channel,
      sound_channel: this.state.settings.sound_channel,
      desktop_channel: this.state.settings.desktop_channel,
    };
    await fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    await this.loadSettings();
    const toast = document.getElementById('alerts-save-toast');
    toast.style.display = 'inline';
    setTimeout(() => toast.style.display = 'none', 2500);
  },
  async testAlert() {
    const res = await fetch('/api/settings/test-alert', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) { alert(data.error || 'Could not send test alert.'); return; }
    this.playAlertSound();
    document.getElementById('email-popup-context').textContent = 'This is a test alert. No incident occurred.';
    document.getElementById('email-popup-recipients').textContent = 'To: ' + data.recipients.join(', ');
    document.getElementById('email-popup-backdrop').style.display = 'flex';
  },
  closeEmailPopup() { document.getElementById('email-popup-backdrop').style.display = 'none'; },

  // ---------------- Analytics ----------------
  async loadAnalytics() {
    const days = document.getElementById('analytics-range').value;
    const res = await fetch(`/api/analytics?days=${days}`);
    const a = await res.json();

    document.getElementById('stat-tiles').innerHTML = `
      <div style="background:#EAF0DE;border-radius:18px;padding:18px 20px"><div style="font-size:12px;color:rgba(41,17,0,0.6);margin-bottom:6px">Total incidents</div><div class="display-font" style="font-size:24px;color:#291100">${a.total_incidents}</div></div>
      <div style="background:#FBF6DE;border-radius:18px;padding:18px 20px"><div style="font-size:12px;color:rgba(41,17,0,0.6);margin-bottom:6px">Avg. confidence</div><div class="display-font" style="font-size:24px;color:#291100">${a.avg_confidence_pct}%</div></div>
      <div style="background:#F3D9C8;border-radius:18px;padding:18px 20px"><div style="font-size:12px;color:rgba(41,17,0,0.6);margin-bottom:6px">Busiest room</div><div class="display-font" style="font-size:18px;color:#291100">${esc(a.busiest_room)}</div></div>
      <div style="background:#D7E3A4;border-radius:18px;padding:18px 20px"><div style="font-size:12px;color:rgba(41,17,0,0.6);margin-bottom:6px">Busiest camera</div><div class="display-font" style="font-size:18px;color:#291100">${esc(a.busiest_camera)}</div></div>
    `;

    const maxDay = Math.max(1, ...a.by_day.map(d => d.count));
    document.getElementById('incidents-by-day-chart').innerHTML = a.by_day.map(d => `
      <div style="flex:1;display:flex;flex-direction:column;align-items:center;height:100%;justify-content:flex-end">
        <div style="width:100%;background:#C9A227;height:${(d.count / maxDay) * 100}%;min-height:${d.count ? '4px' : '0'}"></div>
        <span style="font-size:10px;color:rgba(43,26,8,0.5);margin-top:6px">${d.label}</span>
      </div>`).join('');

    const maxRoom = Math.max(1, ...a.by_room.map(r => r.count));
    document.getElementById('incidents-by-room-chart').innerHTML = a.by_room.map(r => `
      <div>
        <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px"><span>${esc(r.room)}</span><span>${r.count}</span></div>
        <div style="background:#F0ECDA;border-radius:999px;height:8px"><div style="background:#6D8B3E;border-radius:999px;height:8px;width:${(r.count / maxRoom) * 100}%"></div></div>
      </div>`).join('') || '<div style="font-size:13px;color:rgba(43,26,8,0.5)">No incidents in this range.</div>';
  },

  // ---------------- System settings ----------------
  async loadSystemSettings() {
    const res = await fetch('/api/system-settings');
    const s = await res.json();
    this.state.systemSettings = s;

    document.getElementById('model-info').innerHTML = `Path: <code style="font-family:monospace">${esc(s.model_path)}</code><br>Validation F1: 0.8981 &middot; ROC-AUC: 0.9343`;
    document.getElementById('sys-confirm-seconds').value = s.confirm_seconds;
    document.getElementById('sys-motion-threshold').value = s.motion_threshold;
    document.getElementById('sys-buffer').value = s.buffer_seconds;
    document.getElementById('sys-post-event').value = s.post_event_seconds;
    document.getElementById('sys-retention-days').value = s.retention_days;
    document.getElementById('sys-fp-retention-days').value = s.false_positive_retention_days;

    const isAdmin = this.state.isAdmin;
    ['sys-confirm-seconds', 'sys-motion-threshold', 'sys-buffer', 'sys-post-event', 'sys-retention-days', 'sys-fp-retention-days'].forEach(id => {
      document.getElementById(id).disabled = !isAdmin;
    });
    document.getElementById('detection-defaults-note').style.display = isAdmin ? 'none' : 'block';
    document.getElementById('upload-model-btn').style.display = isAdmin ? 'inline-block' : 'none';
    document.getElementById('upload-model-note').style.display = isAdmin ? 'none' : 'block';
    document.getElementById('save-system-btn').style.display = isAdmin ? 'inline-block' : 'none';
    document.getElementById('retention-card').style.display = isAdmin ? 'block' : 'none';
    document.getElementById('invites-card').style.display = isAdmin ? 'block' : 'none';
    document.getElementById('caregiver-rooms-card').style.display = isAdmin ? 'block' : 'none';
    if (isAdmin) { this.loadInvites(); this.loadCaregiverRooms(); this.loadCleanupStatus(); }

    const health = document.getElementById('camera-health-list');
    health.innerHTML = this.state.cameras.map(cam => `
      <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(43,26,8,0.08);font-size:13px">
        <span>${esc(cam.id)} &middot; ${esc(cam.room)}</span>
        <span style="color:${cam.liveStatus === 'Live' ? '#405221' : '#7A2E12'}">${cam.liveStatus}</span>
      </div>`).join('');
  },

  // ---------------- Invites (admin only; server also enforces this) ----------------
  async loadInvites() {
    const res = await fetch('/api/invites');
    if (!res.ok) return;
    const invites = await res.json();
    const list = document.getElementById('pending-invites-list');
    list.innerHTML = invites.map(inv => `
      <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(43,26,8,0.08)">
        <span>${esc(inv.email)} &middot; ${esc(inv.role)} &middot; invited by ${esc(inv.invited_by)}</span>
        <button class="pill-btn pill-btn-outline" style="padding:4px 12px;font-size:11px" onclick="Haven.revokeInvite('${esc(inv.email)}')">Revoke</button>
      </div>`).join('') || '<div style="font-size:13px;color:rgba(43,26,8,0.5)">No pending invites.</div>';
    // Keep tokens around client-side only long enough to revoke by email;
    // the list endpoint deliberately never returns tokens, so revoke below
    // re-fetches with the token from the create response where available.
    this._pendingInvites = invites;
  },
  async createInvite() {
    const email = document.getElementById('invite-email').value.trim();
    const name  = document.getElementById('invite-name').value.trim();
    const role  = document.getElementById('invite-role').value;
    const rooms = document.getElementById('invite-rooms').value.split(',').map(r => r.trim()).filter(Boolean);
    const box = document.getElementById('invite-link-box');
    const errBox = document.getElementById('invite-error');
    box.style.display = 'none';
    errBox.style.display = 'none';
    if (!email) { errBox.textContent = 'Email is required.'; errBox.style.display = 'block'; return; }

    const res = await fetch('/api/invites', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, name, role, assigned_rooms: rooms }),
    });
    const data = await res.json();
    if (!res.ok) { errBox.textContent = data.error || 'Could not create invite.'; errBox.style.display = 'block'; return; }

    const roomNote = role === 'admin' ? 'sees every room (admin)' : (rooms.length ? `rooms: ${rooms.join(', ')}` : 'NO rooms assigned yet -- they will see nothing until you assign some here');
    box.innerHTML = `Invite link for <strong>${esc(data.email)}</strong> (expires in 7 days, ${esc(roomNote)}). Copy and send this to them:<br><code style="word-break:break-all">${esc(data.link)}</code>`;
    box.style.display = 'block';
    document.getElementById('invite-email').value = '';
    document.getElementById('invite-name').value = '';
    document.getElementById('invite-rooms').value = '';
    this.loadInvites();
  },

  // ---------------- Caregiver room access (admin only) ----------------
  async loadCaregiverRooms() {
    const res = await fetch('/api/caregivers');
    if (!res.ok) return;
    const caregivers = await res.json();
    const list = document.getElementById('caregiver-rooms-list');
    list.innerHTML = caregivers.map(u => {
      const isAdminAcct = u.role === 'admin';
      const rooms = (u.assigned_rooms || []).join(', ');
      const inputId = `rooms-${esc(u.email).replace(/[^a-zA-Z0-9]/g, '_')}`;
      return `
      <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid rgba(43,26,8,0.08)">
        <div style="min-width:0">
          <div style="font-weight:600">${esc(u.name || u.email)} <span style="font-weight:400;color:rgba(43,26,8,0.5);font-size:12px">${esc(u.role)}</span></div>
          <div style="font-size:12px;color:rgba(43,26,8,0.5)">${esc(u.email)}</div>
        </div>
        ${isAdminAcct
          ? `<span style="font-size:12px;color:rgba(43,26,8,0.5);flex:none">Sees every room</span>`
          : `<div style="display:flex;gap:8px;align-items:center;flex:none">
               <input id="${inputId}" type="text" value="${esc(rooms)}" placeholder="Ward A, Ward B" class="pill-input" style="width:220px;border-radius:12px;font-size:12px;${rooms ? '' : 'border-color:#C15A1C'}">
               <button class="pill-btn pill-btn-outline" style="padding:4px 12px;font-size:11px" onclick="Haven.saveCaregiverRooms('${esc(u.email)}','${inputId}')">Save</button>
             </div>`}
      </div>`;
    }).join('') || '<div style="font-size:13px;color:rgba(43,26,8,0.5)">No accounts yet.</div>';
  },
  async saveCaregiverRooms(email, inputId) {
    const rooms = document.getElementById(inputId).value.split(',').map(r => r.trim()).filter(Boolean);
    const res = await fetch(`/api/caregivers/${encodeURIComponent(email)}/rooms`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ assigned_rooms: rooms }),
    });
    if (res.ok) this.loadCaregiverRooms();
  },
  async revokeInvite(email) {
    // The dashboard never stores raw tokens beyond the create-invite
    // response, so revoking by email round-trips through the pending list
    // the server already returned (which also omits tokens) -- ask the
    // server to look the invite up and delete it in one step instead.
    const res = await fetch(`/api/invites/by-email/${encodeURIComponent(email)}`, { method: 'DELETE' });
    if (res.ok) this.loadInvites();
  },
  async saveSystemSettings() {
    if (!this.state.isAdmin) return;
    const payload = {
      confirm_seconds: parseInt(document.getElementById('sys-confirm-seconds').value, 10),
      motion_threshold: parseFloat(document.getElementById('sys-motion-threshold').value),
      buffer_seconds: parseInt(document.getElementById('sys-buffer').value, 10),
      post_event_seconds: parseInt(document.getElementById('sys-post-event').value, 10),
      retention_days: parseInt(document.getElementById('sys-retention-days').value, 10),
      false_positive_retention_days: parseInt(document.getElementById('sys-fp-retention-days').value, 10),
    };
    const res = await fetch('/api/system-settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    if (!res.ok) { alert('Could not save (administrator access required).'); return; }
    const toast = document.getElementById('system-save-toast');
    toast.style.display = 'inline';
    setTimeout(() => toast.style.display = 'none', 3000);
  },

  // ---------------- Retention / cleanup (admin only) ----------------
  async loadCleanupStatus() {
    const res = await fetch('/api/admin/cleanup-status');
    if (!res.ok) return;
    const s = await res.json();
    const el = document.getElementById('cleanup-status');
    el.textContent = s.ran_at
      ? `Last run: ${s.ran_at} · deleted ${s.deleted_events} incident(s), ${s.deleted_clips} clip(s)`
      : 'Not run yet since this server started.';
  },
  async runCleanupNow() {
    if (!this.state.isAdmin) return;
    const el = document.getElementById('cleanup-status');
    el.textContent = 'Running…';
    const res = await fetch('/api/admin/run-cleanup', { method: 'POST' });
    if (!res.ok) { el.textContent = 'Could not run cleanup (administrator access required).'; return; }
    this.loadCleanupStatus();
  },
  uploadModel() {
    if (!this.state.isAdmin) return;
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.pt';
    input.onchange = async () => {
      if (!input.files.length) return;
      const fd = new FormData();
      fd.append('model', input.files[0]);
      const res = await fetch('/api/system-settings/upload-model', { method: 'POST', body: fd });
      const data = await res.json();
      alert(res.ok ? data.note : (data.error || 'Upload failed.'));
      if (res.ok) this.loadSystemSettings();
    };
    input.click();
  },
};

function esc(str) {
  const div = document.createElement('div');
  div.textContent = str === undefined || str === null ? '' : String(str);
  return div.innerHTML;
}

document.addEventListener('DOMContentLoaded', () => Haven.init());
