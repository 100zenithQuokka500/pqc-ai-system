"""
main.py — PQC-AI Sentiment API (v3).

New additions over v2:
  • GET  /keys/{mode}          — returns server public keys for client-side encapsulation
  • POST /predict/encrypted    — full end-to-end encrypted request + response path
  • POST /predict              — original plaintext route retained for benchmarking

End-to-end encrypted channel flow
──────────────────────────────────
  1. Client calls GET /keys/post_quantum (or /hybrid, /classical).
  2. Client encapsulates an ephemeral shared secret against the server's KEM public key
     → produces (kem_ciphertext, shared_secret).
  3. Client AES-GCM encrypts the request payload with shared_secret.
  4. Client POSTs /predict/encrypted with {kem_ciphertext, nonce, ciphertext, sensitivity}.
  5. Server:
       a. Policy engine selects algorithm (may differ from the mode used for key exchange;
          we always use the PQ crypto objects for decapsulation here for simplicity).
       b. Decapsulates kem_ciphertext → shared_secret.
       c. Decrypts request payload.
       d. Runs DistilBERT inference.
       e. Encrypts the JSON result with a FRESH server-side AES key.
       f. Encapsulates that fresh key under the server's own KEM public key so the
          client can later demonstrate decapsulation (demo-mode; a real deployment
          would use the CLIENT'S public key for the response path).
       g. Returns EncryptedPredictResponse.
  6. Client decapsulates response_kem_ct → response AES key → decrypts result.
"""

import json
import time
import uuid
import logging
import os

import psutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_model.classifier import predict as ai_predict, is_model_loaded, _get_pipeline
from crypto.classical import ClassicalCrypto
from crypto.post_quantum import PostQuantumCrypto, validate_encapsulation_key
from crypto.hybrid import HybridCrypto
from policy.engine import PolicyEngine
from api.schemas import (
    CryptoMode, SensitivityLevel,
    PredictRequest, PredictResponse,
    EncryptedPredictRequest, EncryptedPredictResponse,
    KeysResponse, HealthResponse, MetricsResponse,
)

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import oqs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

START_TIME    = time.time()
REQUEST_COUNT = 0

classical_crypto  = ClassicalCrypto()
pqc_crypto        = PostQuantumCrypto()
hybrid_crypto     = HybridCrypto()
policy_engine     = PolicyEngine()

CRYPTO_MAP = {
    CryptoMode.CLASSICAL:    classical_crypto,
    CryptoMode.POST_QUANTUM: pqc_crypto,
    CryptoMode.HYBRID:       hybrid_crypto,
}


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_pipeline()          # warm up DistilBERT
    logger.info("DistilBERT warmed up; server ready.")
    yield
    logger.info("Shutting down PQC-AI API.")


app = FastAPI(
    title="PQC-AI Sentiment API",
    version="3.0.0",
    description="Adaptive post-quantum cryptography with end-to-end encrypted channel.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Latency-recording middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def record_latency(request: Request, call_next):
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    if getattr(request.state, "used_policy", False):
        policy_engine.record_request_outcome(duration_ms)
    response.headers["X-Request-Time-Ms"] = str(round(duration_ms, 2))
    return response

# ---------------------------------------------------------------------------
# DEBUG ONLY — exposes server's KEM secret key for the demonstration client.
# Remove this endpoint before production or security review.
# ---------------------------------------------------------------------------
@app.get("/debug/kem_secret_key_hex")
async def debug_kem_secret():
    """Return the server's KEM secret key in hex. DEVELOPMENT ONLY."""
    return {"kem_secret_key_hex": pqc_crypto.kem_secret_key.hex()}

# ===========================================================================
# ROUTE 1 — Health
# ===========================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=is_model_loaded(),
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_percent=psutil.virtual_memory().percent,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


# ===========================================================================
# ROUTE 2 — Public-key distribution (NEW)
# ===========================================================================

@app.get("/keys/{mode}", response_model=KeysResponse)
async def get_keys(mode: CryptoMode):
    """
    Return the server's public key material for the requested encryption mode.

    The client uses these keys to perform client-side key encapsulation before
    sending an encrypted request to POST /predict/encrypted.

    For ML-KEM keys, the returned hex string is 1184 bytes (Kyber-768),
    satisfying FIPS 203 §7.2 Check 1 (type check).
    """
    if mode == CryptoMode.POST_QUANTUM:
        info = pqc_crypto.get_public_key_info()
        return KeysResponse(
            mode=mode,
            kem_public_key_hex=info["kem_public_key"].hex(),
            sig_public_key_hex=info["sig_public_key"].hex(),
            kem_algo=info["kem_algo"],
            sig_algo=info["sig_algo"],
        )
    elif mode == CryptoMode.HYBRID:
        info = hybrid_crypto.get_public_key_info()
        return KeysResponse(
            mode=mode,
            kem_public_key_hex=info["kem_public_key"].hex(),
            sig_public_key_hex=info["sig_public_key"].hex(),
            rsa_public_key_pem=info["rsa_public_key_pem"].decode(),
            kem_algo=info["kem_algo"],
            sig_algo=info["sig_algo"],
        )
    else:  # classical
        info = classical_crypto.get_public_key_info()
        return KeysResponse(
            mode=mode,
            rsa_public_key_pem=info["rsa_public_key_pem"].decode(),
        )


# ===========================================================================
# ROUTE 3 — End-to-end encrypted predict (NEW)
# ===========================================================================

@app.post("/predict/encrypted", response_model=EncryptedPredictResponse)
async def predict_encrypted(request: Request, body: EncryptedPredictRequest):
    """
    End-to-end encrypted sentiment prediction.

    Request  : client-encrypted payload (Kyber KEM ciphertext + AES-GCM ciphertext)
    Response : server-encrypted result  (fresh AES-GCM, key encapsulated under PQ KEM)

    See module docstring for the full 6-step protocol.
    """
    start_total = time.perf_counter()
    request_id  = body.request_id or str(uuid.uuid4())[:8]

    # --- Step 5a: policy selects algorithm ---
    decision = policy_engine.select_algorithm(
        sensitivity=body.sensitivity.value,
        payload_size=len(body.ciphertext_hex) // 2,   # hex → bytes
    )
    chosen_mode = CryptoMode(decision["algorithm"])

    # --- Step 5b: decapsulate kem_ciphertext → shared_secret ---
    if body.kem_ciphertext_hex is None:
        raise HTTPException(
            status_code=400,
            detail="kem_ciphertext_hex is required for encrypted predict."
        )

    kem_ciphertext   = bytes.fromhex(body.kem_ciphertext_hex)
    request_nonce    = bytes.fromhex(body.nonce_hex)
    request_ct       = bytes.fromhex(body.ciphertext_hex)

    try:
        with oqs.KeyEncapsulation(pqc_crypto.KEM_ALGO, pqc_crypto.kem_secret_key) as kem:
            shared_secret = kem.decap_secret(kem_ciphertext)
        aes_key = shared_secret[:32]

        # --- Step 5c: decrypt request payload ---
        try:
            plaintext_bytes = AESGCM(aes_key).decrypt(request_nonce, request_ct, None)
        except Exception:
            raise HTTPException(status_code=400, detail="AES-GCM decryption failed — bad key or tampered ciphertext.")

        request_data = json.loads(plaintext_bytes.decode("utf-8"))
        text = request_data.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Decrypted payload missing 'text' field.")

    finally:
        # Zero sensitive key material (Gap 2 best-effort)
        if isinstance(aes_key, bytearray):
            for i in range(len(aes_key)): aes_key[i] = 0

    # --- Step 5d: AI inference ---
    try:
        ai_result = ai_predict(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # --- Step 5e-f: encrypt response with a fresh server-side key ---
    #
    # The server generates a new ephemeral AES key and encapsulates it under
    # its own Kyber-768 public key.  The client can decapsulate it using the
    # server's secret key material (available in the demo client below).
    #
    # In a production system: replace pqc_crypto.kem_public_key with the
    # CLIENT'S KEM public key, which the client would supply during registration.

    response_payload = json.dumps(ai_result).encode("utf-8")
    encrypted_response = pqc_crypto.encrypt(response_payload)

    total_ms = (time.perf_counter() - start_total) * 1000
    request.state.used_policy = True

    return EncryptedPredictResponse(
        response_nonce_hex=encrypted_response["nonce"].hex(),
        response_ciphertext_hex=encrypted_response["ciphertext"].hex(),
        response_kem_ct_hex=encrypted_response["kem_ciphertext"].hex(),
        crypto_mode=chosen_mode,
        policy_score=decision["score"],
        policy_reasoning=decision["reasoning"],
        total_time_ms=round(total_ms, 2),
        request_id=request_id,
    )


# ===========================================================================
# ROUTE 4 — Original plaintext predict (unchanged, for benchmarking)
# ===========================================================================

@app.post("/predict", response_model=PredictResponse)
async def predict_plaintext(request: Request, body: PredictRequest):
    start_total = time.perf_counter()
    decision = policy_engine.select_algorithm(
        sensitivity=body.sensitivity.value,
        payload_size=len(body.text.encode("utf-8")),
    )
    chosen_mode = CryptoMode(decision["algorithm"])

    try:
        ai_result = ai_predict(body.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Encrypt (measure overhead; result not returned in plaintext mode)
    CRYPTO_MAP[chosen_mode].encrypt(str(ai_result).encode("utf-8"))

    total_ms = (time.perf_counter() - start_total) * 1000
    request.state.used_policy = True

    return PredictResponse(
        label=ai_result["label"],
        score=ai_result["score"],
        inference_time_ms=ai_result["inference_time_ms"],
        total_time_ms=round(total_ms, 2),
        crypto_mode=chosen_mode,
        policy_score=decision["score"],
        policy_reasoning=decision["reasoning"],
        request_id=body.request_id or str(uuid.uuid4())[:8],
    )


# ===========================================================================
# ROUTE 5 — Metrics / Policy
# ===========================================================================

@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    stats = policy_engine.get_stats()
    return MetricsResponse(
        total_requests=REQUEST_COUNT,
        avg_latency_ms=stats.get("avg_latency_ms", 0),
        current_cpu_percent=psutil.cpu_percent(interval=0.1),
        algorithm_distribution=stats.get("distribution", {}),
        q_table_updates=stats.get("q_table_updates", {}),
        replay_buffer_sizes=stats.get("buffer_sizes", {}),
    )


@app.get("/policy/stats")
async def policy_stats():
    return policy_engine.get_stats()


@app.get("/policy/history")
async def policy_history():
    return policy_engine.get_history()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
