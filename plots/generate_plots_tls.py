import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = 'plots/output/paper'
os.makedirs(OUTPUT_DIR, exist_ok=True)

loads = [15, 35, 55, 80]
lat_http = []
lat_tls = []

for u in loads:
    # HTTP
    df_h = pd.read_csv(f'benchmark/results/v3_final/adaptive_{u}_stats.csv')
    row_h = df_h[df_h['Name'] == 'Aggregated']
    lat_http.append(row_h['Average Response Time'].values[0])
    
    # TLS
    df_t = pd.read_csv(f'benchmark/results/tls_final/adaptive_{u}_stats.csv')
    row_t = df_t[df_t['Name'] == 'Aggregated']
    lat_tls.append(row_t['Average Response Time'].values[0])

x = np.arange(len(loads))
width = 0.35
plt.figure(figsize=(8,5))
plt.bar(x - width/2, lat_http, width, label='HTTP', color='#0072B2')
plt.bar(x + width/2, lat_tls, width, label='TLS (HTTPS)', color='#CC79A7')
plt.xticks(x, [f'{u} users' for u in loads])
plt.ylabel('Average Latency (ms)')
plt.title('Latency Comparison: HTTP vs HTTPS')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_http_vs_tls.pdf'), dpi=300)
print('Saved: fig7_http_vs_tls.pdf')