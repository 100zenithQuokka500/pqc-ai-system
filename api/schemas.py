"""
schemas.py — Pydantic models for the PQC-AI Sentiment API.

Additions for real end-to-end encrypted channel:
  • KeysResponse   — public key material returned by GET /keys
  • EncryptedPredictRequest  — client sends pre-encapsulated shared key + text
  • EncryptedPredictResponse — server returns AES-GCM encrypted payload
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Existing enums
# ---------------------------------------------------------------------------

class SensitivityLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class CryptoMode(str, Enum):
    CLASSICAL    = "classical"
    HYBRID       = "hybrid"
    POST_QUANTUM = "post_quantum"


# ---------------------------------------------------------------------------
# Original plaintext request / response (unchanged, for compatibility)
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text:        str            = Field(..., min_length=1, max_length=512)
    sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM
    request_id:  Optional[str] = None


class PredictResponse(BaseModel):
    label:            str
    score:            float
    inference_time_ms: float
    total_time_ms:    float
    crypto_mode:      CryptoMode
    policy_score:     float
    policy_reasoning: str
    request_id:       Optional[str]


# ---------------------------------------------------------------------------
# E2E key-exchange schemas
# ---------------------------------------------------------------------------

class KeysResponse(BaseModel):
    """
    Returned by GET /keys.

    The client uses this to perform client-side encapsulation so that only
    the server (holding the matching secret key) can recover the shared AES key.

    Fields
    ------
    mode                : algorithm mode the keys belong to
    kem_public_key_hex  : Kyber-768 public key, hex-encoded
                          (1184 bytes for Kyber-768; type-check per FIPS 203 §7.2)
    sig_public_key_hex  : ML-DSA-44 verification key, hex-encoded
    rsa_public_key_pem  : RSA-2048 public key in PEM format (hybrid/classical only)
    kem_algo            : KEM algorithm identifier string
    sig_algo            : signature algorithm identifier string
    """
    mode:               CryptoMode
    kem_public_key_hex: Optional[str] = None   # None for classical-only
    sig_public_key_hex: Optional[str] = None
    rsa_public_key_pem: Optional[str] = None   # None for PQ-only
    kem_algo:           Optional[str] = None
    sig_algo:           Optional[str] = None


class EncryptedPredictRequest(BaseModel):
    """
    Sent by the client to POST /predict/encrypted.

    The client has already:
      1. Called GET /keys to obtain the server's public keys.
      2. Generated an ephemeral shared secret and encapsulated it.
      3. Encrypted the plaintext request JSON with AES-GCM using that secret.

    Fields
    ------
    sensitivity         : data sensitivity level (drives policy + reward)
    request_id          : optional idempotency token
    kem_ciphertext_hex  : Kyber KEM ciphertext (client encapsulation output), hex
    rsa_enc_key_hex     : RSA-encrypted AES key (hybrid/classical mode), hex
    nonce_hex           : AES-GCM nonce, hex (12 bytes)
    ciphertext_hex      : AES-GCM encrypted payload, hex
                          Payload format: UTF-8 JSON {"text": "..."}
    """
    sensitivity:        SensitivityLevel = SensitivityLevel.MEDIUM
    request_id:         Optional[str]    = None
    kem_ciphertext_hex: Optional[str]    = None
    rsa_enc_key_hex:    Optional[str]    = None
    nonce_hex:          str              = Field(..., min_length=24, max_length=24)
    ciphertext_hex:     str              = Field(..., min_length=2)


class EncryptedPredictResponse(BaseModel):
    """
    Returned by POST /predict/encrypted.

    The AI result is encrypted with a fresh server-side AES key that is itself
    encapsulated/encrypted under the server's own public key so the client can
    recover it (using the server's matching secret key, obtained via /keys).

    In a fully-deployed system the client would hold its own public key pair
    and the server would encrypt toward the CLIENT'S key.  For this single-server
    demo the server re-uses its own keys for the response path, which is
    sufficient to demonstrate the E2E encrypted channel without a second
    public-key exchange round-trip.

    Fields
    ------
    response_nonce_hex        : AES-GCM nonce for the encrypted result, hex
    response_ciphertext_hex   : AES-GCM encrypted JSON result, hex
    response_kem_ct_hex       : Kyber KEM ciphertext encapsulating the response AES key
    response_rsa_enc_key_hex  : RSA-encrypted response AES key (classical/hybrid)
    crypto_mode               : algorithm chosen by the policy engine
    policy_score              : Q-value of the selected action
    policy_reasoning          : human-readable decision explanation
    total_time_ms             : wall-clock time for the whole request
    request_id                : echoed from the request
    """
    response_nonce_hex:       str
    response_ciphertext_hex:  str
    response_kem_ct_hex:      Optional[str] = None
    response_rsa_enc_key_hex: Optional[str] = None
    crypto_mode:              CryptoMode
    policy_score:             float
    policy_reasoning:         str
    total_time_ms:            float
    request_id:               Optional[str]


# ---------------------------------------------------------------------------
# Existing health / metrics models (unchanged)
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:          str
    model_loaded:    bool
    cpu_percent:     float
    memory_percent:  float
    uptime_seconds:  float


class MetricsResponse(BaseModel):
    total_requests:         int
    avg_latency_ms:         float
    current_cpu_percent:    float
    algorithm_distribution: dict
    q_table_updates:        dict     # now per-sensitivity
    replay_buffer_sizes:    dict     # now per-sensitivity
