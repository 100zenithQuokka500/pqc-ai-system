"""
classical.py — RSA-2048 (OAEP) + AES-256-GCM.

No FIPS 203 gaps apply to this module directly, but we add:
  • public_key_pem() — needed by HybridCrypto for the /keys endpoint.
  • Gap 2 mitigation: AES keys held in bytearray and zeroed after use.
"""

import os
import time
import logging

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

_OAEP = padding.OAEP(
    mgf=padding.MGF1(hashes.SHA256()),
    algorithm=hashes.SHA256(),
    label=None,
)


def _zero(buf: bytearray) -> None:
    for i in range(len(buf)):
        buf[i] = 0


class ClassicalCrypto:
    ALGORITHM_NAME = "RSA-2048+AES-256-GCM"

    def __init__(self) -> None:
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def public_key_pem(self) -> bytes:
        """Return RSA public key in PEM format (needed by hybrid key exchange)."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    # ------------------------------------------------------------------
    # Encrypt / Decrypt
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: bytes) -> dict:
        start = time.perf_counter()
        aes_key_ba: bytearray | None = None
        try:
            aes_key_ba = bytearray(os.urandom(32))
            nonce = os.urandom(12)
            ciphertext = AESGCM(bytes(aes_key_ba)).encrypt(nonce, plaintext, None)
            encrypted_key = self.public_key.encrypt(bytes(aes_key_ba), _OAEP)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "algorithm": self.ALGORITHM_NAME,
                "ciphertext": ciphertext,
                "encrypted_key": encrypted_key,
                "nonce": nonce,
                "encrypt_ms": round(elapsed_ms, 3),
            }
        finally:
            if aes_key_ba is not None:
                _zero(aes_key_ba)

    def decrypt(self, pkg: dict) -> dict:
        start = time.perf_counter()
        aes_key_ba: bytearray | None = None
        try:
            raw = self.private_key.decrypt(pkg["encrypted_key"], _OAEP)
            aes_key_ba = bytearray(raw)
            plaintext = AESGCM(bytes(aes_key_ba)).decrypt(pkg["nonce"], pkg["ciphertext"], None)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"plaintext": plaintext, "decrypt_ms": round(elapsed_ms, 3)}
        finally:
            if aes_key_ba is not None:
                _zero(aes_key_ba)

    def get_public_key_info(self) -> dict:
        return {
            "rsa_public_key_pem": self.public_key_pem(),
            "algorithm": self.ALGORITHM_NAME,
        }
