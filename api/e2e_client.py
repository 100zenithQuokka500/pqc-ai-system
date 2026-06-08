"""
e2e_client.py — Demonstration client for the E2E encrypted channel.

Runs the full 6-step protocol:
  1. GET /keys/post_quantum   → obtain server KEM + signature public keys
  2. FIPS 203 §7.2 validate  → type-check the received KEM key (client side)
  3. Encapsulate              → (kem_ciphertext, shared_secret)
  4. Encrypt request payload  → AES-GCM(shared_secret, {"text": ...})
  5. POST /predict/encrypted  → receive encrypted response
  6. Decapsulate response key → decrypt and print the AI result

Usage:
    python api/e2e_client.py --host http://localhost:8000 --text "Great product!"
"""

import argparse
import json
import os
import sys

import requests
import oqs
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ---------------------------------------------------------------------------
# FIPS 203 §7.2 client-side key validation (mirrors server implementation)
# ---------------------------------------------------------------------------

_KYBER768_K       = 3
_Q                = 3329
_EK_BYTE_LEN      = 384 * _KYBER768_K + 32   # 1184 bytes


def _byte_decode_12(b: bytes) -> list[int]:
    coeffs = []
    for i in range(0, len(b) - 1, 3):
        b0, b1, b2 = b[i], b[i + 1], b[i + 2]
        coeffs.append(b0 | ((b1 & 0x0F) << 8))
        coeffs.append((b1 >> 4) | (b2 << 4))
    return coeffs


def _byte_encode_12(coeffs: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(coeffs) - 1, 2):
        a, b_ = coeffs[i] & 0xFFF, coeffs[i + 1] & 0xFFF
        out.append(a & 0xFF)
        out.append((a >> 8) | ((b_ & 0x0F) << 4))
        out.append(b_ >> 4)
    return bytes(out)


def client_validate_encapsulation_key(ek: bytes) -> None:
    """Client-side FIPS 203 §7.2 checks (same logic as server)."""
    if len(ek) != _EK_BYTE_LEN:
        raise ValueError(
            f"[CLIENT] FIPS 203 §7.2 Check 1 FAIL: "
            f"expected {_EK_BYTE_LEN} bytes, got {len(ek)}"
        )
    t = ek[: 384 * _KYBER768_K]
    coefficients = _byte_decode_12(t)
    for idx, c in enumerate(coefficients):
        if c >= _Q:
            raise ValueError(
                f"[CLIENT] FIPS 203 §7.2 Check 2 FAIL: "
                f"coefficient[{idx}] = {c} ≥ q={_Q}"
            )
    re_encoded = _byte_encode_12(coefficients)
    if re_encoded != t:
        raise ValueError("[CLIENT] FIPS 203 §7.2 Check 2 FAIL: round-trip mismatch.")
    print("[CLIENT] FIPS 203 §7.2: KEM public key validated ✓")


# ---------------------------------------------------------------------------
# E2E client
# ---------------------------------------------------------------------------

class E2EClient:
    """
    Stateless client demonstrating the PQC-AI E2E encrypted channel.

    The server's KEM secret key is NOT held by this client; decapsulation of
    the response key is done by directly calling the server's /keys endpoint
    which (in this demo) returns the same key pair used for both request and
    response directions.  In a real deployment the response would be encrypted
    toward the CLIENT'S public key.
    """

    KEM_ALGO = "Kyber768"

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()

    # ------------------------------------------------------------------
    # Step 1 — Fetch server public keys
    # ------------------------------------------------------------------

    def fetch_keys(self, mode: str = "post_quantum") -> dict:
        resp = self.session.get(f"{self.base_url}/keys/{mode}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Steps 2–4 — Validate, encapsulate, encrypt request
    # ------------------------------------------------------------------

    def prepare_request(
        self,
        text: str,
        sensitivity: str,
        kem_public_key_hex: str,
    ) -> tuple[dict, bytes]:
        """
        Returns (request_body_dict, shared_secret).
        The caller must keep shared_secret to decrypt the response (if needed).
        """
        kem_public_key = bytes.fromhex(kem_public_key_hex)

        # Step 2: FIPS 203 §7.2 client-side validation
        client_validate_encapsulation_key(kem_public_key)

        # Step 3: Kyber-768 encapsulation
        with oqs.KeyEncapsulation(self.KEM_ALGO) as kem:
            kem_ciphertext, shared_secret = kem.encap_secret(kem_public_key)

        # Step 4: AES-GCM encrypt the request payload
        aes_key = shared_secret[:32]
        nonce   = os.urandom(12)
        payload = json.dumps({"text": text}).encode("utf-8")
        ciphertext = AESGCM(aes_key).encrypt(nonce, payload, None)

        body = {
            "sensitivity":       sensitivity,
            "kem_ciphertext_hex": kem_ciphertext.hex(),
            "nonce_hex":          nonce.hex(),
            "ciphertext_hex":     ciphertext.hex(),
        }
        return body, shared_secret

    # ------------------------------------------------------------------
    # Step 5 — POST /predict/encrypted
    # ------------------------------------------------------------------

    def post_encrypted(self, body: dict) -> dict:
        resp = self.session.post(
            f"{self.base_url}/predict/encrypted",
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Step 6 — Decapsulate response key and decrypt AI result
    # ------------------------------------------------------------------

    def decrypt_response(
        self,
        encrypted_response: dict,
        server_kem_secret_key_hex: str,
    ) -> dict:
        """
        In this demo the server's secret key is fetched from a separate debug
        endpoint.  In production the client would use its OWN secret key because
        the server would encrypt toward the client's public key.

        Here we demonstrate the mechanics of the full channel by recovering the
        server's response using the (demo-accessible) server secret.
        """
        response_kem_ct = bytes.fromhex(encrypted_response["response_kem_ct_hex"])
        response_nonce  = bytes.fromhex(encrypted_response["response_nonce_hex"])
        response_ct     = bytes.fromhex(encrypted_response["response_ciphertext_hex"])

        kem_secret_key = bytes.fromhex(server_kem_secret_key_hex)

        with oqs.KeyEncapsulation(self.KEM_ALGO, kem_secret_key) as kem:
            shared_secret = kem.decap_secret(response_kem_ct)

        aes_key   = shared_secret[:32]
        plaintext = AESGCM(aes_key).decrypt(response_nonce, response_ct, None)
        return json.loads(plaintext.decode("utf-8"))

    # ------------------------------------------------------------------
    # Convenience: full round-trip
    # ------------------------------------------------------------------

    def run(self, text: str, sensitivity: str = "high") -> None:
        print(f"\n{'='*60}")
        print(f"PQC-AI E2E Encrypted Channel Demo")
        print(f"{'='*60}")
        print(f"Input text : {text!r}")
        print(f"Sensitivity: {sensitivity}")

        # 1. Fetch keys
        print("\n[1/6] Fetching server public keys...")
        keys = self.fetch_keys("post_quantum")
        print(f"      KEM algo : {keys['kem_algo']}")
        print(f"      Sig algo : {keys['sig_algo']}")
        print(f"      KEM key  : {keys['kem_public_key_hex'][:32]}... ({len(keys['kem_public_key_hex'])//2} bytes)")

        # 2–4. Validate, encapsulate, encrypt
        print("\n[2/6] FIPS 203 §7.2 — validating KEM public key...")
        print("[3/6] Encapsulating shared secret (Kyber-768)...")
        print("[4/6] Encrypting request payload (AES-256-GCM)...")
        body, _shared_secret = self.prepare_request(
            text=text,
            sensitivity=sensitivity,
            kem_public_key_hex=keys["kem_public_key_hex"],
        )

        # 5. POST encrypted request
        print("\n[5/6] Sending encrypted request to POST /predict/encrypted...")
        encrypted_resp = self.post_encrypted(body)
        print(f"      Policy decision : {encrypted_resp['crypto_mode']}")
        print(f"      Policy reasoning: {encrypted_resp['policy_reasoning']}")
        print(f"      Total time      : {encrypted_resp['total_time_ms']:.1f} ms")
        print(f"      Response ct     : {encrypted_resp['response_ciphertext_hex'][:32]}... (encrypted)")

        # 6. Decrypt response (demo: fetch server secret from debug endpoint)
        print("\n[6/6] Decapsulating response key and decrypting AI result...")
        debug = self.session.get(f"{self.base_url}/debug/kem_secret_key_hex", timeout=5)
        if debug.status_code == 200:
            server_secret_hex = debug.json()["kem_secret_key_hex"]
            result = self.decrypt_response(encrypted_resp, server_secret_hex)
            print(f"\n✓  Decrypted AI result:")
            print(f"   Label            : {result['label']}")
            print(f"   Score            : {result['score']}")
            print(f"   Inference time   : {result['inference_time_ms']} ms")
        else:
            print("   (Debug endpoint not available; encrypted response received correctly.)")
            print(f"   Encrypted result (hex): {encrypted_resp['response_ciphertext_hex'][:64]}...")

        print(f"\n{'='*60}")
        print("E2E channel demonstration complete.")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Optional debug endpoint — add this to main.py for the demo client above
# ---------------------------------------------------------------------------
#
# @app.get("/debug/kem_secret_key_hex")
# async def debug_kem_secret(x_debug_token: str = Header(None)):
#     """
#     DEVELOPMENT ONLY — exposes the server's KEM secret key so the demo
#     client can demonstrate full response decryption.
#     REMOVE THIS ENDPOINT BEFORE PRODUCTION DEPLOYMENT.
#     """
#     if x_debug_token != os.environ.get("DEBUG_TOKEN", ""):
#         raise HTTPException(status_code=403, detail="Forbidden")
#     return {"kem_secret_key_hex": pqc_crypto.kem_secret_key.hex()}
#
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PQC-AI E2E Encrypted Client Demo")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--text", default="The product quality is absolutely outstanding.")
    parser.add_argument("--sensitivity", choices=["low", "medium", "high"], default="high")
    args = parser.parse_args()

    client = E2EClient(base_url=args.host)
    client.run(text=args.text, sensitivity=args.sensitivity)
