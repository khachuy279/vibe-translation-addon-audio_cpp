"""Module tạo và quản lý chứng chỉ SSL tự ký cho Backend WSS / HTTPS."""

import datetime
import ipaddress
import os
from pathlib import Path
from typing import Tuple

from backend.utils.logger import get_logger

logger = get_logger("utils.ssl")

BACKEND_DIR = Path(__file__).resolve().parent.parent


def ensure_ssl_certificates() -> Tuple[str, str]:
    """Đảm bảo file chứng chỉ SSL (cert.pem, key.pem) tồn tại, tự sinh nếu thiếu.

    Returns:
        Tuple (cert_path, key_path)
    """
    cert_path = BACKEND_DIR / "cert.pem"
    key_path = BACKEND_DIR / "key.pem"

    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    logger.info("Đang tạo chứng chỉ SSL tự ký cho localhost (WSS/HTTPS)...", extra={"module_tag": "CORE"})

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    # Khởi tạo khóa RSA 2048-bit
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

    # Hạn sử dụng 10 năm
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

    with open(tmp_cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    os.replace(tmp_key_path, key_path)
    os.replace(tmp_cert_path, cert_path)

    logger.info(f"Đã tạo chứng chỉ SSL thành công: {cert_path}", extra={"module_tag": "CORE"})
    return str(cert_path), str(key_path)
