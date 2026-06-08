import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crypto.classical import ClassicalCrypto
from crypto.post_quantum import PostQuantumCrypto
from crypto.hybrid import HybridCrypto
import time, json, statistics
PAYLOAD = b'Sentiment analysis request: The movie was absolutely wonderful!'
ITERATIONS = 100

def benchmark_one(name, crypto_obj):
    enc_times, dec_times = [], []
    for _ in range(ITERATIONS):
        pkg = crypto_obj.encrypt(PAYLOAD)
        enc_times.append(pkg['encrypt_ms'])
        result = crypto_obj.decrypt(pkg)
        dec_times.append(result['decrypt_ms'])
    return {
        'algorithm': name,
        'iterations': ITERATIONS,
        'enc_avg_ms': round(statistics.mean(enc_times), 3),
        'enc_stdev_ms': round(statistics.stdev(enc_times), 3),
        'enc_min_ms': round(min(enc_times), 3),
        'enc_max_ms': round(max(enc_times), 3),
        'dec_avg_ms': round(statistics.mean(dec_times), 3),
        'dec_stdev_ms': round(statistics.stdev(dec_times), 3),
        'payload_bytes': len(PAYLOAD),
    }

if __name__ == '__main__':
    classical = ClassicalCrypto()
    pqc = PostQuantumCrypto()
    hybrid = HybridCrypto()
    print('Running crypto benchmarks...')
    results = [
        benchmark_one('Classical (RSA+AES)', classical),
        benchmark_one('Post-Quantum (K+D+AES)', pqc),
        benchmark_one('Hybrid (RSA+K+D+AES)', hybrid),
    ]
    print(json.dumps(results, indent=2))
    # Save for later
    with open('benchmark/results/crypto_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved to benchmark/results/crypto_baseline.json')
