"""SSL certificate helper for backend_audio_cpp.

Ensures valid certificates exist for HTTPS / WSS communication on localhost.
"""

import datetime
import ipaddress
import logging
import os
from pathlib import Path
from typing import Tuple

logger = logging.getLogger("backend_audio_cpp.ssl")

_BACKEND_DIR = Path(__file__).resolve().parent.parent


def ensure_ssl_certificates() -> Tuple[str, str]:
    """Ensure SSL certificates exist, reusing or generating self-signed pair.

    Returns:
        Tuple of (cert_path, key_path)
    """
    cert_path = _BACKEND_DIR / "cert.pem"
    key_path = _BACKEND_DIR / "key.pem"

    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    # Check if backend_cpp_old has certs that Firefox already trusted
    old_cert = _BACKEND_DIR.parent / "backend_cpp_old" / "cert.pem"
    old_key = _BACKEND_DIR.parent / "backend_cpp_old" / "key.pem"
    if old_cert.exists() and old_key.exists():
        try:
            cert_path.write_bytes(old_cert.read_bytes())
            key_path.write_bytes(old_key.read_bytes())
            logger.info(f"Reused existing SSL certificates from {old_cert}")
            return str(cert_path), str(key_path)
        except Exception:
            pass

    logger.info("🔑 Generating self-signed SSL certificate for localhost (WSS/HTTPS)...")

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "VN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VibeTranslation"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    san = x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv6Address("::1")),
    ])

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )

    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    logger.info(f"✅ Created self-signed certificate: {cert_path}")
    return str(cert_path), str(key_path)
