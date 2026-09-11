"""SSL certificate helper for backend_cpp.

Automatically creates self-signed certificates if none are found, ensuring
WSS (WebSocket Secure) and HTTPS work seamlessly with Firefox/Chrome extensions.
"""

import datetime
import ipaddress
import logging
import os
from pathlib import Path
from typing import Tuple

logger = logging.getLogger("backend_cpp.ssl")

_BACKEND_CPP_DIR = Path(__file__).resolve().parent.parent


def ensure_ssl_certificates() -> Tuple[str, str]:
    """Ensure SSL certificates exist, generating a self-signed pair if missing.

    Returns:
        Tuple of (cert_path, key_path)
    """
    cert_path = _BACKEND_CPP_DIR / "cert.pem"
    key_path = _BACKEND_CPP_DIR / "key.pem"


    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    logger.info("🔑 Generating self-signed SSL certificate for localhost (WSS/HTTPS)...")

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    # Generate 2048-bit RSA key
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

    # Valid for 10 years
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

    # Write private key atomically via temporary file
    tmp_key_path = key_path.with_suffix(".pem.tmp")
    tmp_cert_path = cert_path.with_suffix(".pem.tmp")

    with open(tmp_key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    if os.name != "nt":
        try:
            os.chmod(tmp_key_path, 0o600)
        except Exception:
            pass

    # Write certificate atomically via temporary file
    with open(tmp_cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    # Atomically rename to final destinations to prevent race conditions / corrupted cert pairs
    os.replace(tmp_key_path, key_path)
    os.replace(tmp_cert_path, cert_path)

    if os.name != "nt":
        try:
            os.chmod(key_path, 0o600)
        except Exception:
            pass

    logger.info(f"✅ Created self-signed certificate: {cert_path}")
    return str(cert_path), str(key_path)
