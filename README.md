# Hybrid Implementation and Evaluation of Classical and Post-Quantum Cryptographic Mechanisms in AI-Based Web Services

## Authors

**Himangi Tripathy, Nikita Tosh, Annanya Nayak, Reetu Kanungo, Rishi Raj Gupta**
*Students, Department of Computer Science and Engineering*

**Prof. Amit Kumar Kar** *(Faculty Supervisor)*
*Department of Computer Science and Engineering*

Siksha 'O' Anusandhan (Deemed to be) University, Bhubaneswar, Odisha, India

📧 himangi1tripathy@gmail.com · nikitatosh@gmail.com · annanyanayak@gmail.com · reetukanungo@gmail.com · rishi765449@gmail.com

---

A FastAPI-based sentiment analysis service that combines **DistilBERT inference** with an **adaptive post-quantum cryptography (PQC) policy engine**. The policy engine uses per-sensitivity Double Q-Learning to dynamically select the most appropriate cryptographic algorithm for each request based on real-time CPU load, latency, and data sensitivity level.

---

## Overview

The system exposes a REST API for sentiment prediction with three cryptographic modes:

| Mode | Implementation |
|------|----------------|
| `classical` | RSA-2048 + AES-256-GCM |
| `post_quantum` | ML-KEM-768 (Kyber-768) + ML-DSA-44 + AES-256-GCM (FIPS 203/204 compliant) |
| `hybrid` | RSA-2048 + Kyber-768 with HKDF key derivation (SP 800-56C Rev.2) |

An RL-based **Policy Engine** selects the algorithm at runtime — without requiring the client to specify one — based on sensitivity level (`low`, `medium`, `high`), current CPU utilisation, and observed request latency.

---

## Project Structure

```
updated_code/
├── ai_model/
│   ├── classifier.py          # DistilBERT sentiment inference wrapper
│   └── test_classifier.py     # Unit tests for classifier
├── api/
│   ├── main.py                # FastAPI application, routes, middleware
│   └── schemas.py             # Pydantic request/response models
├── crypto/
│   ├── classical.py           # RSA-2048 + AES-256-GCM
│   ├── post_quantum.py        # ML-KEM-768 + ML-DSA-44 (FIPS 203/204)
│   ├── hybrid.py              # RSA + Kyber with HKDF combination
│   └── benchmark_crypto.py   # Standalone crypto benchmarking script
├── policy/
│   ├── engine.py              # PolicyEngine — per-sensitivity Double Q-tables
│   ├── qtable.py              # DoubleQTable implementation
│   ├── replay_buffer.py       # Experience replay buffer
│   └── monitor.py             # System CPU/latency monitor
├── benchmark/
│   ├── locustfile.py          # Locust load test definition
│   └── results/               # Benchmark result JSONs and CSVs
├── plots/
│   ├── generate_plots.py      # Matplotlib plot generation (paper figures)
│   └── output/paper/          # Generated PDF figures (fig1–fig7)
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── nginx/nginx.conf        # TLS termination via nginx reverse proxy
```

---

## API Endpoints

### `GET /health`
Returns server health status including model load state, CPU/memory usage, and uptime.

### `GET /keys/{mode}`
Returns the server's public key material for the requested crypto mode (`classical`, `hybrid`, or `post_quantum`). Used by clients to perform key encapsulation before sending an encrypted request.

### `POST /predict`
Plaintext sentiment prediction. The policy engine selects the crypto algorithm internally; encryption is performed but the result is not returned (intended for latency benchmarking).

**Request body:**
```json
{
  "text": "This product is excellent.",
  "sensitivity": "medium",
  "request_id": "optional-string"
}
```

**Response:**
```json
{
  "label": "POSITIVE",
  "score": 0.9998,
  "inference_time_ms": 42.1,
  "total_time_ms": 55.3,
  "crypto_mode": "post_quantum",
  "policy_score": 0.712,
  "policy_reasoning": "[MEDIUM] Double-Q selected post_quantum | cpu=12.3% lat=55.3ms ε=0.0821"
}
```

### `POST /predict/encrypted`
Full end-to-end encrypted prediction. The client encrypts the request payload using Kyber-768 KEM + AES-GCM; the server decrypts, runs inference, and returns an encrypted response.

**6-step protocol:**
1. Client fetches server public keys via `GET /keys/post_quantum`
2. Client encapsulates an ephemeral shared secret → `(kem_ciphertext, shared_secret)`
3. Client encrypts request payload with AES-GCM using `shared_secret`
4. Client POSTs `{kem_ciphertext_hex, nonce_hex, ciphertext_hex, sensitivity}`
5. Server decapsulates → decrypts → runs DistilBERT → encrypts response with a fresh AES key encapsulated under its own KEM public key
6. Client decapsulates `response_kem_ct` → recovers response AES key → decrypts result

### `GET /metrics`
Aggregate request statistics, algorithm distribution, Q-table update counts, and replay buffer sizes.

### `GET /policy/stats` / `GET /policy/history`
Detailed RL policy introspection: per-sensitivity reward histories, epsilon trajectories, and decision logs (used for paper plot generation).

---

## Policy Engine

The `PolicyEngine` maintains **one DoubleQTable per sensitivity level** (`low`, `medium`, `high`) to eliminate cross-sensitivity Q-value interference.

**Reward function:**

```
reward = w_sec * security_score(alg)
       - w_lat * min(latency_ms / 300, 1)
       - w_cpu * min(cpu_pct / 80, 1)
```

Sensitivity weights:

| Sensitivity | Security | Latency | CPU |
|-------------|----------|---------|-----|
| low | 0.25 | 0.45 | 0.30 |
| medium | 0.40 | 0.36 | 0.24 |
| high | 0.65 | 0.21 | 0.14 |

For `high` sensitivity requests:
- A dynamic latency-aware bonus is added when `post_quantum` is selected: `bonus = 0.15 × (1 − latency/300)`
- A fixed penalty of `−0.20` is applied when `classical` or `hybrid` is selected

Adaptive re-exploration: if the trailing reward drops more than 1 standard deviation below the rolling mean, epsilon is reset to `0.20`.

---

## Cryptographic Compliance Notes

- **FIPS 203 §7.2** — `validate_encapsulation_key()` in `post_quantum.py` enforces both mandatory pre-encapsulation checks (type check: `len(ek) == 1184`; modulus check: ByteDecode₁₂ ∘ ByteEncode₁₂ round-trip).
- **FIPS 203 §3.3** — sensitive key material is held in `bytearray` buffers and zeroed in `finally` blocks. Note: Python's GC does not guarantee deterministic zeroing; a C/Rust extension is required for production-grade memory hygiene.
- **SP 800-56C Rev.2 §4** — the hybrid mode replaces the original XOR+SHA-256 key combination with HKDF-SHA-256 (`info = b"PQC-AI-Hybrid-v1"`) for domain separation and formal security guarantees.

> ⚠️ **Development note:** The `GET /debug/kem_secret_key_hex` endpoint exposes the server's KEM secret key. **Remove this endpoint before any production deployment or security review.**

---

## Setup & Running

### Requirements

- Python 3.11
- `liboqs` (Open Quantum Safe library) — installed via `pip install oqs`
- Docker + Docker Compose (for containerised deployment)

### Local (without Docker)

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker Compose (with TLS via nginx)

```bash
cd docker
docker compose up --build
```

This starts three services:
- **pqc-api** — FastAPI app on internal port 8000 (not exposed directly)
- **nginx** — TLS termination proxy on ports 443 (HTTPS) and 80
- **locust** — Load testing UI on port 8089

SSL certificates are expected at `docker/nginx/ssl/server.crt` and `server.key`. The included certificate is self-signed and intended for development only.

---

## Load Testing

Locust is pre-configured with weighted tasks reflecting a realistic traffic mix:

| Task | Weight | Description |
|------|--------|-------------|
| `predict_medium` | 5 | Standard medium-sensitivity prediction |
| `predict_high` | 2 | High-sensitivity — policy should prefer `post_quantum` |
| `predict_low` | 1 | Low-sensitivity — policy may choose `classical` |
| `health_check` | 1 | Liveness probe |

Run via the Docker Compose setup or directly:

```bash
locust -f benchmark/locustfile.py --host https://localhost
```

Results are saved to `benchmark/results/` as CSV and JSON files (stats, history, policy decisions).

---

## Plot Generation

Paper figures (fig1–fig7) are generated from benchmark result JSON files:

```bash
python plots/main.py
```

Output PDFs are written to `plots/output/paper/`. Figures cover PQ algorithm distribution vs load, RL learning curves, per-sensitivity algorithm distributions over time, CPU scatter plots, stacked distributions, epsilon decay, and HTTP vs TLS latency comparison.

---

## Model

Sentiment classification uses [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) from HuggingFace Transformers, running on CPU. The model is pre-downloaded during the Docker build step to avoid latency on the first request.
