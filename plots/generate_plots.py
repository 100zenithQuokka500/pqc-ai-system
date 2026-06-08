import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

OUTPUT_DIR = 'plots/output/paper'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
LOAD_NAMES = ['15 users', '35 users', '55 users', '80 users']
LOAD_VALUES = [15, 35, 55, 80]
BASE_PATH = '/home/user/final_year_project/pqc_ai_project/benchmark/results/v3_final'

# Colors (professional, colorblind‑friendly)
COLORS = {
    'classical': '#E69F00',      # orange
    'hybrid': '#56B4E9',         # light blue
    'post_quantum': '#009E73',   # teal
    'reward': '#0072B2',         # dark blue
}

# ------------------------------------------------------------------
# Helper: load policy JSON
# ------------------------------------------------------------------
def load_policy(load):
    fname = f'adaptive_{load}_policy.json'
    with open(os.path.join(BASE_PATH, fname)) as f:
        return json.load(f)

# ------------------------------------------------------------------
# 1. Post‑Quantum % vs Load (Fig 1)
# ------------------------------------------------------------------
pq_pct = []
for load in LOAD_VALUES:
    data = load_policy(load)
    decisions = data['decisions']
    last_1000 = decisions[-1000:] if len(decisions) >= 1000 else decisions
    pq = sum(1 for d in last_1000 if d['algorithm'] == 'post_quantum')
    pq_pct.append(pq / len(last_1000) * 100)

plt.figure(figsize=(6,5))
x = np.arange(len(LOAD_NAMES))
bars = plt.bar(x, pq_pct, color=COLORS['post_quantum'], edgecolor='black', linewidth=1)
plt.ylim(0,100)
plt.ylabel('Post‑Quantum Selection (%)', fontsize=12)
plt.xlabel('Load Level', fontsize=12)
plt.title('Post‑Quantum Usage vs Load (HTTP)', fontsize=14)
plt.xticks(x, LOAD_NAMES, fontsize=10)
for i, v in enumerate(pq_pct):
    plt.text(i, v+1, f'{v:.1f}%', ha='center', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_pq_vs_load.pdf'), dpi=300)
plt.close()
print('Saved: fig1_pq_vs_load.pdf')

# ------------------------------------------------------------------
# 2. Learning Curve (55 users, medium sensitivity) – Fig 2
# ------------------------------------------------------------------
data_55 = load_policy(55)
rewards = data_55['rewards']['medium']          # choose medium sensitivity
window = 50
if len(rewards) >= window:
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10,5))
    plt.plot(range(window-1, len(rewards)), smoothed, color=COLORS['reward'], linewidth=2)
    plt.xlabel('Request Number', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Learning Curve (55 users, Medium Sensitivity)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_learning_curve.pdf'), dpi=300)
    plt.close()
    print('Saved: fig2_learning_curve.pdf')

# ------------------------------------------------------------------
# 3. Algorithm Distribution Over Time (55 users) – Fig 3
# ------------------------------------------------------------------
decisions = data_55['decisions']
window = 100
algorithms = ['classical', 'hybrid', 'post_quantum']
plt.figure(figsize=(12,5))
for alg in algorithms:
    pcts = []
    for i in range(window, len(decisions)+1, window):
        batch = decisions[i-window:i]
        cnt = sum(1 for d in batch if d['algorithm'] == alg)
        pcts.append(cnt / window * 100)
    plt.plot(pcts, label=alg.capitalize(), color=COLORS[alg], linewidth=2)
plt.ylim(0,100)
plt.xlabel(f'Window ({window} requests)', fontsize=12)
plt.ylabel('Selection Frequency (%)', fontsize=12)
plt.title('Algorithm Distribution Over Time (55 users)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_distribution_over_time.pdf'), dpi=300)
plt.close()
print('Saved: fig3_distribution_over_time.pdf')

# ------------------------------------------------------------------
# 4. CPU vs Algorithm Scatter (55 users) – Fig 4
# ------------------------------------------------------------------
plt.figure(figsize=(10,6))
for alg, color in COLORS.items():
    if alg not in algorithms:
        continue
    pts = [d for d in decisions if d['algorithm'] == alg]
    if pts:
        x_vals = list(range(len(pts)))
        y_vals = [d['cpu'] for d in pts]
        plt.scatter(x_vals[:200], y_vals[:200], c=color, alpha=0.5, s=15, label=alg.capitalize())
plt.xlabel('Request Index (first 200 per algorithm)', fontsize=12)
plt.ylabel('CPU% at Decision Time', fontsize=12)
plt.title('CPU Load vs Algorithm (55 users)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_cpu_scatter.pdf'), dpi=300)
plt.close()
print('Saved: fig4_cpu_scatter.pdf')

# ------------------------------------------------------------------
# 5. Stacked Bar Chart per Load (Last 1000 decisions) – Fig 5
# ------------------------------------------------------------------
data_per_load = {}
for load in LOAD_VALUES:
    data = load_policy(load)
    decisions = data['decisions']
    last_1000 = decisions[-1000:] if len(decisions) >= 1000 else decisions
    counts = defaultdict(int)
    for d in last_1000:
        counts[d['algorithm']] += 1
    total = len(last_1000)
    data_per_load[load] = {alg: counts[alg]/total*100 for alg in algorithms}

fig, ax = plt.subplots(figsize=(8,5))
x = np.arange(len(LOAD_NAMES))
width = 0.6
bottom = np.zeros(len(LOAD_NAMES))
for alg in algorithms:
    vals = [data_per_load[load][alg] for load in LOAD_VALUES]
    ax.bar(x, vals, width, label=alg.capitalize(), bottom=bottom, color=COLORS[alg], edgecolor='black', linewidth=0.5)
    bottom += np.array(vals)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_xlabel('Load Level', fontsize=12)
ax.set_title('Algorithm Distribution (Last 1000 Decisions)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(LOAD_NAMES, fontsize=10)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_stacked_distribution.pdf'), dpi=300)
plt.close()
print('Saved: fig5_stacked_distribution.pdf')

# ------------------------------------------------------------------
# 6. Epsilon Decay (55 users, medium sensitivity) – Fig 6
# ------------------------------------------------------------------
epsilons = data_55['epsilons']['medium']
plt.figure(figsize=(10,5))
plt.plot(epsilons, color=COLORS['reward'], linewidth=2)
plt.axhline(y=0.05, color='red', linestyle='--', label='Min epsilon (0.05)')
plt.xlabel('Request Number', fontsize=12)
plt.ylabel('Epsilon', fontsize=12)
plt.title('Exploration Rate Decay (55 users, Medium Sensitivity)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_epsilon_decay.pdf'), dpi=300)
plt.close()
print('Saved: fig6_epsilon_decay.pdf')

# ------------------------------------------------------------------
# 7. Optional: Extract metrics for LaTeX tables
# ------------------------------------------------------------------
# Extract average latency and throughput from CSV files (if needed)
# For completeness, compute per‑sensitivity distribution for 55 users.
from collections import defaultdict as dd
sens_counts = defaultdict(lambda: defaultdict(int))
for d in decisions:
    sens = d.get('sensitivity', 'medium')
    alg = d['algorithm']
    sens_counts[sens][alg] += 1

print("\n=== Per‑sensitivity distribution (55 users) ===")
for sens in ['low', 'medium', 'high']:
    total = sum(sens_counts[sens].values())
    if total == 0: continue
    cl = sens_counts[sens]['classical'] / total * 100
    hy = sens_counts[sens]['hybrid'] / total * 100
    pq = sens_counts[sens]['post_quantum'] / total * 100
    print(f"{sens.capitalize()}: Classical={cl:.1f}%, Hybrid={hy:.1f}%, Post‑Quantum={pq:.1f}%")

print(f"\nAll figures saved in {OUTPUT_DIR}")