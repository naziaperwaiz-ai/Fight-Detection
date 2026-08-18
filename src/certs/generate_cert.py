"""
Generate a self-signed TLS certificate for the Haven dashboard.

Why this exists: Flask's dev server (`app.run()`) speaks plain HTTP by
default. Browsers refuse to run "secure context" APIs -- Notification,
Push, several others -- over plain HTTP, so features like the desktop
alert toggle silently do nothing until the dashboard is served over
HTTPS. A self-signed cert satisfies that browser check even though it
isn't issued by a public certificate authority; the tradeoff is that
each caregiver's browser shows a one-time "this site is not trusted"
warning the first time it connects; see src/certs/README.md.

This is the right tool for a facility's local network. It is NOT a
substitute for a real certificate (Let's Encrypt via a reverse proxy,
or a private network like Tailscale) if this server is ever reachable
from the open internet -- see the "What's the deployer's responsibility"
section of the top-level README.

Usage:
    pip install cryptography   # one-time, if not already installed
    python src/certs/generate_cert.py
    python src/certs/generate_cert.py 192.168.1.42 haven.local   # add extra SANs

Writes cert.pem and key.pem next to this script. Both are gitignored --
see the entry added to .gitignore alongside this file.
"""
import datetime
import ipaddress
import os
import stat
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CERT_DIR = Path(__file__).resolve().parent
CERT_PATH = CERT_DIR / "cert.pem"
KEY_PATH = CERT_DIR / "key.pem"

# Always include localhost/127.0.0.1 so the cert works for local testing.
# Pass this machine's LAN IP (e.g. 192.168.1.42) as an extra argument so
# caregivers' phones/tablets connecting over WiFi get a SAN match too --
# without it, browsers reject the cert for that address even though it's
# self-signed and would otherwise be accepted after the trust warning.
DEFAULT_HOSTS = ["localhost", "127.0.0.1"]


def build_san_list(hosts):
    entries = []
    for h in hosts:
        try:
            entries.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            entries.append(x509.DNSName(h))
    return entries


def main():
    extra_hosts = sys.argv[1:]
    hosts = DEFAULT_HOSTS + extra_hosts

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "haven-dashboard"),
    ])

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        # Self-signed + LAN-only: not subject to the ~398-day cap public CAs
        # must follow, since no browser's trust-store policy applies to a
        # cert nobody but you ever marks as trusted. Long validity here
        # just means less re-generation, not less scrutiny.
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(build_san_list(hosts)), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    KEY_PATH.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    CERT_PATH.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    # write_bytes() creates the file with the process's default mode
    # (governed by umask -- commonly 644, world-readable). That's fine for
    # cert.pem (public by design) but not for key.pem: on a shared
    # facility server, any other local account could otherwise read the
    # private key. Restrict it to owner read/write only. (No-op/no-op-ish
    # on Windows, where this bit doesn't map the same way -- real
    # protection there comes from filesystem ACLs / who has an account on
    # the machine at all.)
    try:
        os.chmod(KEY_PATH, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass

    print(f"Wrote {CERT_PATH}")
    print(f"Wrote {KEY_PATH}")
    print(f"Certificate covers: {', '.join(hosts)}")
    if not extra_hosts:
        print(
            "\nNo extra hostnames/IPs were given, so this cert only matches "
            "localhost/127.0.0.1. If caregivers will reach this dashboard at "
            "this machine's LAN IP (e.g. 192.168.1.42), re-run:\n"
            "    python src/certs/generate_cert.py 192.168.1.42"
        )
    print(
        "\nRestart the dashboard (`py src/main.py`) and it will pick up this "
        "cert automatically. The first connection from each browser will show "
        "a \"not private\" warning -- that's expected for a self-signed cert; "
        "click through it once. See src/certs/README.md for details."
    )


if __name__ == "__main__":
    main()
