"""
hybrid.py — RSA-2048 + Kyber-768 + ML-DSA-44 + AES-256-GCM.

Gap addressed (Gap 4 — Hybrid Key Derivation):
    Original code combined keys via:
        combined = XOR(classical_aes_key, kyber_secret[:32])
        final_key = SHA-256(combined)

    FIPS 203 §3.3 requires: "If further key derivation is needed, the final
    symmetric keys shall be derived … in an approved manner, as specified in
    SP 800-108 and SP 800-56C."

    XOR + raw SHA-256 is NOT listed in SP 800-56C.  We replace it with HKDF
    (RFC 5869, NIST SP 800-56C Rev.2 §4), which provides:
        • Formal security proof (indistinguishable from random given good inputs)
        • Domain separation via the `info` parameter
        • Correct entropy bounds

    The `info` label b'PQC-AI-Hybrid-v1' binds the derived key to this
    protocol version, preventing cross-context key reuse.

Gap 2 (secure memory): Sensitive bytearrays are zeroed in finally-blocks.
"""

import os
import time
import logging

import oqs
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from crypto.classical import ClassicalCrypto
from crypto.post_quantum import PostQuantumCrypto, validate_encapsulation_key, _zero

logger = logging.getLogger(__name__)

# Domain-separation label (binds derived key to this protocol + version)
_HKDF_INFO = b"PQC-AI-Hybrid-v1"
_OAEP_PADDING = padding.OAEP(
    mgf=padding.MGF1(hashes.SHA256()),
    algorithm=hashes.SHA256(),
    label=None,
)


def _derive_key_hkdf(classical_secret: bytes, kyber_secret: bytes) -> bytes:
    """
    SP 800-56C Rev.2 §4 — HKDF-based key combination.

    Input keying material = classical_secret || kyber_secret (concatenated).
    Extract+Expand via HKDF-SHA-256 → 32-byte AES key.

    This replaces the non-standard XOR+SHA-256 of the original implementation.
    """
    ikm = classical_secret + kyber_secret           # combined IKM
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,                                   # HKDF uses zero-salt internally
        info=_HKDF_INFO,
    )
    return hkdf.derive(ikm)


class HybridCrypto:
    """
    Hybrid encryption: RSA-2048 + Kyber-768 key exchange, HKDF key derivation,
    ML-DSA-44 signature, AES-256-GCM encryption.

    Security rationale: an attacker must break BOTH RSA (classical threat) AND
    Kyber (quantum threat) to recover the session key.  HKDF ensures the
    combination provides provable security under standard assumptions.
    """

    ALGORITHM_NAME = "RSA-2048+Kyber768+ML-DSA-44+AES-256-GCM (HKDF)"

    def __init__(self) -> None:
        self._classical = ClassicalCrypto()
        self._pqc = PostQuantumCrypto()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: bytes) -> dict:
        """
        Hybrid encrypt.
        Key derivation now uses HKDF (SP 800-56C) instead of XOR+SHA-256.
        Sensitive intermediates are zeroed on exit.
        """
        start = time.perf_counter()

        # FIPS 203 §7.2 – validate KEM public key before use
        validate_encapsulation_key(self._pqc.kem_public_key)

        classical_secret_ba: bytearray | None = None
        kyber_secret_ba: bytearray | None = None
        final_key_ba: bytearray | None = None

        try:
            # 1. RSA side: generate a random classical secret
            classical_secret_raw = os.urandom(32)
            classical_secret_ba = bytearray(classical_secret_raw)

            # 2. Kyber side: encapsulate against PQC public key
            with oqs.KeyEncapsulation(self._pqc.KEM_ALGO) as kem:
                kem_ct, kyber_secret_raw = kem.encap_secret(self._pqc.kem_public_key)
            kyber_secret_ba = bytearray(kyber_secret_raw)

            # 3. HKDF combination (SP 800-56C) — replaces XOR+SHA-256
            final_key = _derive_key_hkdf(
                bytes(classical_secret_ba),
                bytes(kyber_secret_ba[:32]),
            )
            final_key_ba = bytearray(final_key)

            # 4. AES-256-GCM encryption
            nonce = os.urandom(12)
            ciphertext = AESGCM(bytes(final_key_ba)).encrypt(nonce, plaintext, None)

            # 5. RSA encrypt the classical secret (key transport)
            rsa_enc_key = self._classical.public_key.encrypt(
                bytes(classical_secret_ba), _OAEP_PADDING
            )

            # 6. ML-DSA-44 signature over ciphertext
            with oqs.Signature(self._pqc.SIG_ALGO, self._pqc.sig_secret_key) as sig:
                signature = sig.sign(ciphertext)

            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "algorithm": self.ALGORITHM_NAME,
                "ciphertext": ciphertext,
                "rsa_enc_key": rsa_enc_key,
                "kem_ct": kem_ct,
                "nonce": nonce,
                "signature": signature,
                "encrypt_ms": round(elapsed_ms, 3),
            }
        finally:
            # Gap 2: zero all sensitive intermediates
            for buf in (classical_secret_ba, kyber_secret_ba, final_key_ba):
                if buf is not None:
                    _zero(buf)

    def decrypt(self, pkg: dict) -> dict:
        """
        Hybrid decrypt.
        Key recovery also uses HKDF to mirror the encrypt path.
        Sensitive intermediates are zeroed on exit.
        """
        start = time.perf_counter()

        classical_secret_ba: bytearray | None = None
        kyber_secret_ba: bytearray | None = None
        final_key_ba: bytearray | None = None

        try:
            # 1. Verify ML-DSA-44 signature before decrypting anything
            with oqs.Signature(self._pqc.SIG_ALGO) as sig:
                valid = sig.verify(
                    pkg["ciphertext"], pkg["signature"], self._pqc.sig_public_key
                )
            if not valid:
                raise ValueError("ML-DSA-44 signature verification failed — ciphertext tampered.")

            # 2. RSA decrypt classical secret
            classical_secret_raw = self._classical.private_key.decrypt(
                pkg["rsa_enc_key"], _OAEP_PADDING
            )
            classical_secret_ba = bytearray(classical_secret_raw)

            # 3. Kyber decapsulate
            with oqs.KeyEncapsulation(self._pqc.KEM_ALGO, self._pqc.kem_secret_key) as kem:
                kyber_secret_raw = kem.decap_secret(pkg["kem_ct"])
            kyber_secret_ba = bytearray(kyber_secret_raw)

            # 4. HKDF re-derive (must use same IKM order and info label)
            final_key = _derive_key_hkdf(
                bytes(classical_secret_ba),
                bytes(kyber_secret_ba[:32]),
            )
            final_key_ba = bytearray(final_key)

            # 5. AES-GCM decrypt
            plaintext = AESGCM(bytes(final_key_ba)).decrypt(
                pkg["nonce"], pkg["ciphertext"], None
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"plaintext": plaintext, "decrypt_ms": round(elapsed_ms, 3)}
        finally:
            for buf in (classical_secret_ba, kyber_secret_ba, final_key_ba):
                if buf is not None:
                    _zero(buf)

    def get_public_key_info(self) -> dict:
        """Return public key material needed by the client for the hybrid key exchange."""
        return {
            "rsa_public_key_pem": self._classical.public_key_pem(),
            "kem_public_key": self._pqc.kem_public_key,
            "sig_public_key": self._pqc.sig_public_key,
            "kem_algo": self._pqc.KEM_ALGO,
            "sig_algo": self._pqc.SIG_ALGO,
            "kdf": "HKDF-SHA256 (SP 800-56C Rev.2 §4)",
        }
