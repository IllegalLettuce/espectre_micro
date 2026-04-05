import pandas as pd
import numpy as np
from pathlib import Path
import glob

# ── Load all eval CSVs ─────────────────────────────────────────────
files = glob.glob('csi_eval_*.csv')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['correct'] = df['correct'].astype(str).str.lower() == 'true'

STRIDE = 8

# ── Per-session traffic estimate ───────────────────────────────────
sessions = df.groupby(['ground_truth', 'door_state']).apply(lambda g: pd.Series({
    'n_predictions'  : len(g),
    'esp32_hz'       : g['esp32_pps'].mean(),
    'avg_payload_B'  : g['payload_bytes'].mean(),
    'kbps'           : (g['esp32_pps'].mean() * g['payload_bytes'].mean()) / 1024,
    'accuracy_pct'   : g['correct'].mean() * 100,
    'avg_conf'       : g['confidence'].mean(),
    'avg_infer_ms'   : g['inference_ms'].mean(),
    'p95_infer_ms'   : g['inference_ms'].quantile(0.95),
    'avg_dropped'    : g['packets_dropped'].mean(),
})).reset_index()

print("\n=== Per-Session Summary ===")
print(sessions.to_string(index=False))

# ── Per door state accuracy ────────────────────────────────────────
door_acc = df.groupby(['door_state', 'ground_truth']).apply(lambda g: pd.Series({
    'n'          : len(g),
    'accuracy'   : g['correct'].mean() * 100,
    'avg_conf'   : g['confidence'].mean(),
})).reset_index()

print("\n=== Accuracy by Door State ===")
print(door_acc.pivot_table(
    index='door_state',
    columns='ground_truth',
    values='accuracy',
    aggfunc='mean'
).round(1).to_string())

# ── Overall NFR5 check ─────────────────────────────────────────────
overall_kbps = (df['esp32_pps'].mean() * df['payload_bytes'].mean()) / 1024
print(f"\n=== NFR5 Traffic ===")
print(f"  Overall avg KB/s : {overall_kbps:.2f}")
print(f"  NFR5 {'✓ PASS' if overall_kbps < 25 else '✗ FAIL'}  (threshold: 25 KB/s)")

# ── Overall inference latency ──────────────────────────────────────
print(f"\n=== Inference Latency (all sessions) ===")
print(f"  Mean : {df['inference_ms'].mean():.2f} ms")
print(f"  p95  : {df['inference_ms'].quantile(0.95):.2f} ms")
print(f"  Max  : {df['inference_ms'].max():.2f} ms")

# ── Overall accuracy ───────────────────────────────────────────────
print(f"\n=== Overall Accuracy by Class ===")
print(df.groupby('ground_truth')['correct'].agg(
    n='count', accuracy=lambda x: f"{x.mean()*100:.1f}%"
).to_string())

sessions.to_csv('eval_summary.csv', index=False)
print("\nSaved: eval_summary.csv")