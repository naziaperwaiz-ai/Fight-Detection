# TLS certificate for the dashboard

The dashboard runs over plain HTTP by default, which is fine for
`curl`/API calls but blocks browser "secure context" features --
right now that's the desktop notification toggle (Notification API);
more may land here later (Push API, etc.). This folder holds a
self-signed certificate that unblocks those features for a facility's
local network.

## Generate one

```bash
pip install cryptography   # one-time
python src/certs/generate_cert.py
```

If caregivers connect over WiFi at this machine's LAN IP rather than
`localhost`, pass it in so the certificate actually matches the address
they use:

```bash
python src/certs/generate_cert.py 192.168.1.42
```

This writes `cert.pem` and `key.pem` into this folder. `src/main.py`
looks for both automatically on startup -- no other config needed.
Both files are gitignored; never commit `key.pem`.

## What to expect in the browser

Every caregiver's browser will show a "this site is not private" (or
similar) warning the first time it connects, because the cert isn't
signed by a public certificate authority -- only by itself. That's
expected, not a sign something's broken. Click through it ("Advanced
-> Proceed") once per browser/device; most browsers remember that
choice for that site afterward.

## When this is *not* the right fix

This is a LAN-appropriate shortcut, not a substitute for a real
certificate. If this server is ever reachable from the open internet,
see the top-level README's "What's the deployer's responsibility"
section -- a self-signed cert there just trains caregivers to click
through security warnings, which is a habit you don't want them to
have. A reverse proxy with a Let's Encrypt certificate, or a private
network like Tailscale, is the correct tool for remote access.

## Rotating / regenerating

Just re-run `generate_cert.py` -- it overwrites both files. Restart the
dashboard afterward. There's no expiry to worry about day-to-day (see
the long validity window in the script's comments), but regenerate any
time the LAN IP changes or you suspect `key.pem` leaked.
