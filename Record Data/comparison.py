
import pandas as pd
import numpy as np

old = pd.read_csv('csi_training_data_baseline_0_0_1_20260228_121640.csv')
new = pd.read_csv('csi_training_data_baseline_0_0_1_20260313_101156_new.csv')

print(f"Old samples: {len(old)}, New samples: {len(new)}")
print(f"Old duration: 45s implied, rate: {len(old)/45:.1f} Hz")
print(f"New duration: 45s implied, rate: {len(new)/45:.1f} Hz")

agg_cols = ['entropy_turb','iqr_turb','variance_turb','skewness','kurtosis',
            'amp_mean','amp_range','amp_std','amp_mean_low','amp_mean_mid','amp_mean_high']

print("\n=== AGG FEATURE COMPARISON (mean ± std) ===")
print(f"{'Feature':<20} {'Old mean':>10} {'Old std':>10} {'New mean':>10} {'New std':>10} {'Δ mean':>10}")
for col in agg_cols:
    if col in old.columns and col in new.columns:
        print(f"{col:<20} {old[col].mean():>10.4f} {old[col].std():>10.4f} {new[col].mean():>10.4f} {new[col].std():>10.4f} {new[col].mean()-old[col].mean():>+10.4f}")

sc_cols_old = [c for c in old.columns if c.startswith('sc_amp_')]
sc_cols_new = [c for c in new.columns if c.startswith('sc_amp_')]
print(f"\nOld sc_amp columns: {len(sc_cols_old)}")
print(f"New sc_amp columns: {len(sc_cols_new)}")

# Compare sc_amp means
old_sc_mean = old[sc_cols_old].mean(axis=0).values
new_sc_mean = new[sc_cols_new].mean(axis=0).values
print(f"\nOld sc_amp overall mean: {old_sc_mean.mean():.4f}")
print(f"New sc_amp overall mean: {new_sc_mean.mean():.4f}")
print(f"Old sc_amp overall std:  {old_sc_mean.std():.4f}")
print(f"New sc_amp overall std:  {new_sc_mean.std():.4f}")
print(f"Max per-subcarrier delta: {np.abs(new_sc_mean - old_sc_mean).max():.4f} (sc_{np.argmax(np.abs(new_sc_mean - old_sc_mean))})")
print(f"Mean per-subcarrier delta: {np.abs(new_sc_mean - old_sc_mean).mean():.4f}")
