# Security Audit — Haven / Fight-Detection Dashboard

**Scope:** `src/` (Flask dashboard, detection pipeline, notification, auth) and `scripts/`.
**Method:** manual review of every route in `dashboard/app.py`, the auth/session layer in `auth/users.py`, all client-side rendering in `dashboard/static/js/dashboard.js`, and config/secrets handling — plus targeted greps for `eval`/`exec`/`shell=True`/`pickle`/hardcoded-credential patterns across the whole tree.
**Not applicable / addendum:** at the time of this audit, all persistence was flat JSON via `json.load`/`json.dump`, so classic SQL injection had no attack surface. Since then, incident events (`dashboard/events_store.py`'s `SqliteEventsStore`) were migrated to SQLite for write-scaling reasons unrelated to this audit — see the README's "Multi-Camera Architecture"/"Key Design Decisions" sections. Every query in that module uses parameterized placeholders (`?`), never string-formatted SQL, so this doesn't reopen a SQL injection surface; noted here only so this line doesn't read as stale/wrong for a future reviewer. Every other store remains flat JSON. Command injection has no attack surface either — no `subprocess`, `os.system`, or `shell=True` calls exist in the reviewed code.

Six findings, ranked by severity within each of your four categories.

---

## 1. Secrets Management

### 1.1 Hardcoded live-looking credentials in a file that has already left the deployment machine
**Severity: Critical**
**File:** `src/detection/config.py`, lines 16–17, 27, 34

```python
EMAIL_SENDER       = "fightdetectionalerts@gmail.com"
EMAIL_APP_PASSWORD = "eyldoxixbhyevhzt"
...
SECRET_KEY         = "01b6d67aac658a0ed7475b68caf2218b100944ad3985a33f69ed43e21c723de3"
...
INTERNAL_API_KEY   = "eb20cb6e049119e30a3b7579e51317823b5c25751f09984c22813fc85a821ec5"
```

**Risk:** `config.py` is correctly listed in `.gitignore`, so it was never meant to be committed — that part of the design is right. But the file exists, with these values filled in, in the project tree that was uploaded for review. `EMAIL_APP_PASSWORD` is exactly the shape of a real 16-character Gmail App Password, and `SECRET_KEY`/`INTERNAL_API_KEY` are exactly the shape of real generated secrets (compare against the placeholder text in `config.example.py`, which says `"your-16-char-app-password"` / `"change-me-to-a-random-..."`). Whoever holds `SECRET_KEY` can forge a valid Flask session cookie for any user, including an admin, without ever authenticating. Whoever holds `INTERNAL_API_KEY` can forge events at `/api/events/add`. Whoever holds the Gmail app password can send mail as that account and, depending on Gmail account settings, potentially read it.

**I need to flag something specific to this conversation:** because `config.py` was present in the project directory, it was included in the two full-project zip files I sent you earlier in this session via `SendUserFile`. If these are real, active credentials, you should treat them as already exposed and rotate the Gmail app password and regenerate both keys now, regardless of anything else in this report.

**Remediation:** don't store real secrets as class attributes in a Python file that lives in the deployment directory at all — pull them from the environment at runtime, so the only thing on disk is a reference, not the value:

```python
# src/detection/config.py
import os

class Config:
    ...
    EMAIL_SENDER       = os.environ.get("HAVEN_EMAIL_SENDER", "")
    EMAIL_APP_PASSWORD = os.environ.get("HAVEN_EMAIL_APP_PASSWORD", "")
    ...
    SECRET_KEY         = os.environ.get("HAVEN_SECRET_KEY", "")
    INTERNAL_API_KEY   = os.environ.get("HAVEN_INTERNAL_API_KEY", "")
```

`dashboard/app.py` already has the right fallback pattern for `SECRET_KEY` (`... or secrets.token_hex(32)` at line 51) — extend that same "never hardcode, generate if absent" posture to `config.py` itself instead of writing real values into a plaintext file.

---

## 2. Input Validation & Injection

### 2.1 Path Traversal → Arbitrary File Write in the model-upload endpoint
**Severity: Critical**
**File:** `src/dashboard/app.py`, lines 941–950

```python
@app.route("/api/system-settings/upload-model", methods=["POST"])
@admin_required
def upload_model():
    if "model" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["model"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Model file must be a .pt file"}), 400
    dest = MODELS_DIR / f.filename
    f.save(str(dest))
```

**Risk:** `f.filename` is the filename the client sent in the upload's `Content-Disposition` header — entirely attacker-controlled, and nothing here calls `secure_filename()` or validates it beyond checking the `.pt` suffix. `endswith(".pt")` does not stop `../`. A request with `filename="../../../src/dashboard/app.py"` (still ends in... wait, it must end in `.pt`, but `filename="../../src/detection/config.py.pt"` would still fail the check since it must literally end in `.pt` — however `filename="../../auth/users.py"` fails the suffix check, so the attacker is constrained to writing files whose name ends in `.pt`. That's still enough: an attacker can write a `.pt` file **anywhere the process has write access**, for example over `outputs/logs/cameras.json.pt`-style siblings, or — more seriously — into `models/../../src/detection/__pycache__/whatever.pt`, or simply drop a malicious `.pt` checkpoint at an arbitrary path that a future manual `MODEL_PATH` change would `torch.load()`. Directory traversal for write access, even suffix-constrained, is a real primitive: at minimum it lets an attacker plant files outside `models/`, exhaust disk in unexpected directories, or overwrite an existing `.pt` file elsewhere in the tree if one exists there.

This is also reachable via CSRF (see finding 3.2 below), which is what pushes it from "an admin could misuse this" to "an admin's browser could be made to do this without the admin knowing," while visiting an unrelated malicious page.

**Remediation:** sanitize the filename and verify the resolved destination actually stays inside `MODELS_DIR`:

```python
from werkzeug.utils import secure_filename

@app.route("/api/system-settings/upload-model", methods=["POST"])
@admin_required
def upload_model():
    if "model" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["model"]
    filename = secure_filename(f.filename or "")
    if not filename.endswith(".pt"):
        return jsonify({"error": "Model file must be a .pt file"}), 400

    dest = (MODELS_DIR / filename).resolve()
    if dest.parent != MODELS_DIR.resolve():
        return jsonify({"error": "Invalid filename"}), 400

    f.save(str(dest))
    s = load_system_settings()
    s["model_path"] = f"models/{filename}"
    save_system_settings(s)
    return jsonify({"status": "ok", "note": "..."})
```

`secure_filename()` alone strips `../` and path separators; the `dest.parent != MODELS_DIR.resolve()` check is a second, independent guard so the endpoint fails closed even if a future Werkzeug version changes what `secure_filename()` considers safe.

### 2.2 Stored XSS via unescaped/under-escaped admin-controlled fields
**Severity: High**
**File:** `src/dashboard/static/js/dashboard.js`, lines 434–435, 451, 871, 924–925

Two distinct gaps, both landing in the same place — an inline `onclick="Haven.someMethod('...')"` attribute built by string interpolation:

```js
// line 434-435 — cam.id is NOT escaped at all
<button ... onclick="Haven.editCamera('${cam.id}')">Edit</button>
<button ... onclick="Haven.requestDeleteCamera('${cam.id}')">Delete</button>
// line 451 — same, for the preview toggle
<button ... onclick="Haven.togglePreview('${cam.id}')">...</button>
// line 871 — inv.email IS run through esc(), but esc() isn't sufficient here (see below)
<button ... onclick="Haven.revokeInvite('${esc(inv.email)}')">Revoke</button>
// line 924-925 — same pattern for a caregiver's room list / email
<input ... value="${esc(rooms)}" ...>
<button ... onclick="Haven.saveCaregiverRooms('${esc(u.email)}','${inputId}')">Save</button>
```

`cam.id` comes straight from `add_camera()` in `app.py` (`data.get("id", ...)`) with **no format validation at all** — any string is accepted. A caregiver's `email` is validated only for containing `"@"` (`auth/users.py`'s `create_caregiver`), and a room name is validated only for being a non-empty string (`set_assigned_rooms`). None of these reject quote characters.

**Why `esc()` (line 999) doesn't fix this even where it's used:**
```js
function esc(str) {
  const div = document.createElement('div');
  div.textContent = str === undefined || str === null ? '' : String(str);
  return div.innerHTML;
}
```
This round-trips through `textContent`/`innerHTML`, which only escapes `&`, `<`, `>` — it does not escape `"` or `'`, because those only need escaping in an *attribute-value* serialization context, and this function is producing plain text-node content. So an admin who sets a camera ID to:
```
CAM-01'); alert(document.cookie); //
```
produces, after interpolation:
```html
<button onclick="Haven.editCamera('CAM-01'); alert(document.cookie); //')">Edit</button>
```
The HTML parser doesn't care about the embedded `'` characters (the attribute is delimited by the outer `"`), so it hands the whole thing to the JS engine as the `onclick` handler body — which is now `Haven.editCamera('CAM-01'); alert(document.cookie); //')`, valid JavaScript that runs `alert(document.cookie)` for **every caregiver who loads the Cameras page**, not just the admin who set it. And critically: even escaping `"` and `'` as HTML entities (`&quot;`/`&#39;`) would *not* fix this, because inline event-handler attributes are HTML-entity-decoded by the browser before being compiled as JavaScript — so an HTML-entity-encoded quote still turns back into a literal quote at the point the JS engine parses it, and still breaks out of the string. The only real fix is to stop building executable JavaScript by string-concatenating untrusted data.

**Risk:** this crosses a real privilege boundary. A compromised or careless admin account (or an admin tricked via the CSRF path in 3.2) can plant a payload that runs in the browser of every caregiver who views Cameras, Invites, or Team Access — stealing their session cookie, silently calling admin-only endpoints as them if they happen to have elevated access, or exfiltrating what rooms/patients they can see.

**Remediation:** stop interpolating data into inline event-handler attributes; bind the id via a `data-*` attribute (still needs `esc()` for the small remaining risk of a value containing `"`, but the value is never treated as JS source, so a bare `'` in it is harmless) and attach the handler in JS with `addEventListener`, not in markup:

```js
// render
el.innerHTML = this.state.cameras.map(cam => `
  <div class="card" data-cam-id="${esc(cam.id)}">
    ...
    <button class="edit-camera-btn" data-cam-id="${esc(cam.id)}">Edit</button>
    <button class="delete-camera-btn" data-cam-id="${esc(cam.id)}">Delete</button>
  </div>
`).join('');

// bind once, outside the render loop (event delegation)
document.getElementById('cameras-list').addEventListener('click', (e) => {
  const editBtn = e.target.closest('.edit-camera-btn');
  if (editBtn) return Haven.editCamera(editBtn.dataset.camId);
  const delBtn = e.target.closest('.delete-camera-btn');
  if (delBtn) return Haven.requestDeleteCamera(delBtn.dataset.camId);
});
```
Apply the same pattern to `togglePreview`, `revokeInvite`, and `saveCaregiverRooms`. This removes the class of bug entirely rather than patching each call site's escaping.

As defense-in-depth (not a substitute for the above), also harden `esc()` itself so any remaining plain-attribute usages (like the `value="${esc(rooms)}"` on line 924, which *is* a genuine HTML-attribute context and does need quote-escaping) are safe:

```js
function esc(str) {
  return String(str === undefined || str === null ? '' : str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
```

---

## 3. Authentication & Authorization Gaps

Route-guard coverage itself is good — I checked every route in `app.py`: `@login_required`/`@admin_required`/`@internal_key_required` are applied consistently, room-scoped access (`can_access_room`) is checked on every camera/incident/clip route including the "deny on ambiguous" defaults for deleted cameras and unparseable clip filenames, invite tokens are compared with `secrets.compare_digest` (timing-safe), and `verify_caregiver` deliberately runs a dummy hash check on a nonexistent email to prevent user enumeration via timing. No IDOR was found. The one real gap is structural, not a missing check:

### 3.1 No CSRF protection anywhere
**Severity: Medium**
**File:** `src/dashboard/app.py` (app-wide — no `CSRFProtect`, no per-form tokens); confirmed by grep across the whole file

**Risk:** every state-changing route is a plain POST/DELETE with no CSRF token requirement. Most of the dashboard's routes are called via `fetch(...)` with `Content-Type: application/json`, which is a "non-simple" content type — the browser requires a CORS preflight before sending it cross-origin, and since this app never sends an `Access-Control-Allow-Origin` for a third-party origin (the wildcard-stripping hook at line 60 only ever removes `*`, it never adds a real one), that preflight fails and the browser never sends the actual request. That incidentally protects the JSON endpoints. It does **not** protect `/api/system-settings/upload-model` (finding 2.1), which accepts `multipart/form-data` — a "simple" content type that a plain cross-site `<form>` can submit without any preflight, and the request is executed server-side even though the browser would block the attacker's page from reading the response. A malicious page an admin merely has open in another tab can silently trigger a file write on the server the moment the admin is logged in.

**Remediation:** add `Flask-WTF`'s `CSRFProtect` (or an equivalent) and require the token on every state-changing route, including the file upload:

```python
from flask_wtf import CSRFProtect

csrf = CSRFProtect(app)
```
```html
<!-- login.html / signup.html -->
<input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
```
For the JSON `fetch()` calls in `dashboard.js`, read the token from a meta tag and send it as a header:
```html
<meta name="csrf-token" content="{{ csrf_token() }}">
```
```js
const CSRF_TOKEN = document.querySelector('meta[name="csrf-token"]').content;
fetch('/api/cameras/add', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'X-CSRFToken': CSRF_TOKEN },
  body: JSON.stringify(payload),
});
```
And exempt only the genuinely server-to-server route that already has its own auth (`/api/events/add`, gated by `internal_key_required`), since that one is never called from a browser session:
```python
csrf.exempt(add_event)
```

---

## 4. Insecure Cryptography / Misconfiguration

### 4.1 Missing security response headers
**Severity: Medium**
**File:** `src/dashboard/app.py` — only the CORS-wildcard-stripping hook exists (lines 60–65); no CSP, `X-Frame-Options`, `X-Content-Type-Options`, or HSTS header is set anywhere

**Risk:** with no `Content-Security-Policy`, the stored-XSS in finding 2.2 has no second layer of defense — a CSP with `script-src 'self'` would have stopped the inline-`onclick` payload from executing even with the bug present. With no `X-Frame-Options`/`frame-ancestors`, the dashboard can be embedded in an attacker's `<iframe>` and clickjacked — e.g., overlay invisible buttons over "Delete camera" or "Mark false positive" and trick a logged-in caregiver into clicking through. With no `X-Content-Type-Options: nosniff`, some older browsers can be coaxed into MIME-sniffing an uploaded file (see 2.1) as something executable.

**Remediation:** add a response hook alongside the existing CORS one:

```python
@app.after_request
def _security_headers(resp):
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "frame-ancestors 'none'"
    )
    return resp
```
(`script-src 'self'` requires removing the inline `onclick` handlers per finding 2.2's remediation first, or the CSP will break the current UI — which is itself a good forcing function to actually do that fix.)

### 4.2 Session cookie not marked `Secure`; dashboard can silently fall back to plaintext HTTP
**Severity: Medium**
**Files:** `src/dashboard/app.py` (no `SESSION_COOKIE_SECURE` set); `src/main.py`, `_ssl_context()`, lines 52–79

**Risk:** `main.py`'s `_ssl_context()` returns `None` — and the app happily serves over plain HTTP — whenever no cert is present at `HAVEN_SSL_CERT`/`HAVEN_SSL_KEY` (this is documented, intentional "fresh checkout still runs" behavior, not a bug in itself). The problem is that Flask's session cookie is never explicitly marked `Secure`, so even on a deployment that *does* have a cert and normally serves HTTPS, the cookie would still be accepted over a plaintext HTTP connection if one is ever reachable (e.g., someone hits the LAN IP on port 5000 directly, or a proxy misconfiguration briefly exposes HTTP). On a plain-HTTP deployment, the session cookie carrying login state for the caregiver dashboard — which shows live patient video — is sent over the network in cleartext, interceptable by anything else on the same WiFi/LAN segment.

**Remediation:** set the cookie flags explicitly, and make the HTTP fallback a loud opt-in rather than a silent default:

```python
# src/dashboard/app.py, near app.secret_key
app.config.update(
    SESSION_COOKIE_SECURE=True,     # cookie is never sent over plain HTTP
    SESSION_COOKIE_HTTPONLY=True,   # already Flask's default, made explicit
    SESSION_COOKIE_SAMESITE="Lax",
)
```
If plain-HTTP local development needs to keep working, gate `SESSION_COOKIE_SECURE` off only when explicitly running in a documented dev mode (e.g., an env var), not as the unconditional default — the production path should never be able to silently downgrade.

---

## Summary

| # | Finding | Severity | Category |
|---|---|---|---|
| 1.1 | Hardcoded credentials in `config.py` (already redistributed this session) | **Critical** | Secrets management |
| 2.1 | Path traversal / arbitrary file write in `upload_model()` | **Critical** | Injection |
| 2.2 | Stored XSS via camera id / email / room name in inline `onclick` attributes | **High** | Injection |
| 3.1 | No CSRF protection (exploitable via the multipart upload endpoint) | **Medium** | Auth/authz |
| 4.1 | Missing CSP / X-Frame-Options / nosniff headers | **Medium** | Crypto/misconfig |
| 4.2 | Session cookie not `Secure`; silent plaintext-HTTP fallback | **Medium** | Crypto/misconfig |

Everything else checked and found solid, worth naming so it's not mistaken for unreviewed: password hashing (Werkzeug PBKDF2, no plaintext storage), timing-safe comparisons for invite tokens and login (including the dummy-hash-check-on-unknown-email pattern), room-scoped authorization on every camera/incident/clip route with deny-on-ambiguous defaults, `torch.load(..., weights_only=True)` guarding both real model-loading paths against pickle deserialization RCE, TLS cert generation with correct key-file permissions (owner-only), and rate limiting on login/signup/test-alert/invite-creation.
