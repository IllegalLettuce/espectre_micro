"""
ESPectre Training Data Verification Script
==========================================
Loads all CSV files from csv_data/, groups them by class + door state,
and flags outlier sessions and potential mislabelled data before training.

Run from the same directory as the training script:
    python verify_training_data.py

Or specify a different data directory:
    python verify_training_data.py --data-dir /path/to/csv_data
"""

import pandas as pd
import numpy as np
import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS — modify these as needed
# ═══════════════════════════════════════════════════════════════════════════════

# Z-score threshold for session-level feature means within a group
# (same class + door state). Sessions deviating beyond this are flagged.
# Lower = stricter. Recommended: 1.5–2.5
Z_SCORE_THRESHOLD = 2.0

# Max relative deviation of a session's per-subcarrier mean fingerprint
# from the group mean fingerprint, as a fraction of the group mean amplitude.
# E.g. 0.40 = session fingerprint deviates > 40% from group average shape.
SC_FINGERPRINT_DEVIATION_THRESHOLD = 0.40

# For Baseline sessions: maximum allowable session-mean variance_turb.
# Baseline should be nearly static — values above this suggest motion
# contamination (robot vacuum, appliances, passing people, etc.).
BASELINE_MAX_VARIANCE_TURB_MEAN = 1.2

# For Baseline sessions: maximum allowable 95th-percentile variance_turb spike.
# Catches sessions where the baseline is mostly quiet but has bursts of motion.
BASELINE_MAX_VARIANCE_TURB_P95 = 3.0

# For motion sessions (Q1/Q2): minimum mean variance_turb.
# A motion session with very low turbulence may have been recorded while
# standing still, or be mislabelled.
MOTION_MIN_VARIANCE_TURB_MEAN = 0.3

# Minimum samples per session. Shorter sessions may not produce enough
# windows for robust training.
MIN_SAMPLES_PER_SESSION = 600

# Maximum acceptable sample rate deviation from the group median (fraction).
# E.g. 0.20 = flag sessions whose rate differs > 50% from group median.
SAMPLE_RATE_DEVIATION_THRESHOLD = 0.50

# Minimum sessions per group (class + door state) to run outlier detection.
# Groups with fewer sessions are reported as under-represented only.
MIN_SESSIONS_FOR_OUTLIER_DETECTION = 2

# Aggregate features used for cross-session comparison
COMPARISON_FEATURES = [
    'entropy_turb', 'iqr_turb', 'variance_turb',
    'amp_mean', 'amp_std',
    'amp_mean_low', 'amp_mean_mid', 'amp_mean_high',
]

# Valid subcarrier indices (802.11n HT20, null subcarriers excluded)
VALID_SC_INDICES = [
    6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
    21,22,23,24,25,26,34,35,36,37,38,39,40,41,42,43
]

# Directory structure mapping
DIRECTORY_MAPPINGS = {
    'baseline':    'Baseline',
    'movement_q1': 'Quadrant_1',
    'movement_q2': 'Quadrant_2',
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

RESET  = '\033[0m'
BOLD   = '\033[1m'
RED    = '\033[91m'
YELLOW = '\033[93m'
GREEN  = '\033[92m'
CYAN   = '\033[96m'
DIM    = '\033[2m'

def c(text, colour): return f"{colour}{text}{RESET}"
def ok(text):   return c(f"  ✓  {text}", GREEN)
def warn(text): return c(f"  ⚠  {text}", YELLOW)
def err(text):  return c(f"  ✗  {text}", RED)
def info(text): return c(f"  ·  {text}", DIM)
def head(text): return c(f"\n{'='*80}\n{text}\n{'='*80}", BOLD)
def sub(text):  return c(f"\n── {text} {'─'*(76-len(text))}", CYAN)


def parse_door_state(filename: str) -> str | None:
    """
    Extract door state string (e.g. '0_1_0') from a CSV filename.

    Supports formats:
      csi_training_data_baseline_0_1_0_20260313_113804.csv
      csi_training_data_movement_hallway_1_0_1_20260313_115351.csv
      csi_training_data_movement_stairs_0_1_1_20260313_114729.csv
    """
    # Match exactly three 0/1 tokens followed by an 8-digit date
    match = re.search(r'_([01])_([01])_([01])_\d{8}_', filename)
    if match:
        return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
    return None


def compute_session_stats(df: pd.DataFrame, filename: str) -> dict:
    """
    Compute per-session summary statistics used for outlier detection.
    Returns a flat dict of scalar statistics.
    """
    sc_cols = [f'sc_amp_{i}' for i in VALID_SC_INDICES if f'sc_amp_{i}' in df.columns]

    # Timing
    try:
        timestamps = pd.to_datetime(df['timestamp'])
        duration   = (timestamps.max() - timestamps.min()).total_seconds()
        rate       = len(df) / duration if duration > 0 else 0
    except Exception:
        duration = 0
        rate     = 0

    stats = {
        'filename':    filename,
        'n_samples':   len(df),
        'duration_s':  round(duration, 1),
        'rate_hz':     round(rate, 2),
    }

    # Aggregate feature means and percentiles
    for feat in COMPARISON_FEATURES:
        if feat in df.columns:
            stats[f'{feat}_mean'] = float(df[feat].mean())
            stats[f'{feat}_std']  = float(df[feat].std())
            stats[f'{feat}_p95']  = float(df[feat].quantile(0.95))
        else:
            stats[f'{feat}_mean'] = np.nan
            stats[f'{feat}_std']  = np.nan
            stats[f'{feat}_p95']  = np.nan

    # Per-subcarrier mean fingerprint (normalised by overall mean amplitude)
    if sc_cols:
        sc_vals              = df[sc_cols].values.astype('float32')
        global_amp_mean      = sc_vals.mean() + 1e-6
        sc_mean_norm         = sc_vals.mean(axis=0) / global_amp_mean
        stats['sc_norm_mean']  = sc_mean_norm     # array, used for fingerprint comparison
        stats['amp_global']    = float(global_amp_mean)
    else:
        stats['sc_norm_mean'] = np.zeros(len(VALID_SC_INDICES))
        stats['amp_global']   = 0.0

    return stats


def check_group_outliers(group_stats: list[dict], class_name: str,
                          door_state: str) -> list[str]:
    """
    Given a list of per-session stat dicts for a single (class, door_state)
    group, return a list of warning strings for any detected outliers.
    """
    warnings = []
    n = len(group_stats)
    if n < MIN_SESSIONS_FOR_OUTLIER_DETECTION:
        return warnings

    # ── 1. Z-score outlier detection on aggregate feature means ──────────────
    for feat in COMPARISON_FEATURES:
        key   = f'{feat}_mean'
        vals  = np.array([s[key] for s in group_stats if not np.isnan(s.get(key, np.nan))])
        fnames = [s['filename'] for s in group_stats if not np.isnan(s.get(key, np.nan))]
        if len(vals) < 2:
            continue
        mu, sd = vals.mean(), vals.std()
        if sd < 1e-9:
            continue
        for fname, v in zip(fnames, vals):
            z = abs(v - mu) / sd
            if z > Z_SCORE_THRESHOLD:
                direction = 'HIGH' if v > mu else 'LOW'
                warnings.append(
                    f"{fname}: '{feat}' mean is {direction} "
                    f"(value={v:.4f}, group_mean={mu:.4f}, z={z:.2f})"
                )

    # ── 2. Spatial fingerprint deviation ─────────────────────────────────────
    sc_fingerprints = np.array([s['sc_norm_mean'] for s in group_stats])
    group_fp        = sc_fingerprints.mean(axis=0)
    for s, fp in zip(group_stats, sc_fingerprints):
        # Mean absolute deviation as a fraction of group mean fingerprint magnitude
        denom     = np.abs(group_fp).mean() + 1e-9
        deviation = np.abs(fp - group_fp).mean() / denom
        if deviation > SC_FINGERPRINT_DEVIATION_THRESHOLD:
            warnings.append(
                f"{s['filename']}: subcarrier fingerprint deviates "
                f"{deviation*100:.1f}% from group mean "
                f"(threshold={SC_FINGERPRINT_DEVIATION_THRESHOLD*100:.0f}%) "
                f"— possible channel/antenna shift or mislabel"
            )

    # ── 3. Sample rate consistency ────────────────────────────────────────────
    rates  = np.array([s['rate_hz'] for s in group_stats])
    median_rate = np.median(rates)
    for s, r in zip(group_stats, rates):
        if median_rate > 0 and abs(r - median_rate) / median_rate > SAMPLE_RATE_DEVIATION_THRESHOLD:
            warnings.append(
                f"{s['filename']}: sample rate {r:.1f} Hz deviates "
                f">{SAMPLE_RATE_DEVIATION_THRESHOLD*100:.0f}% from group median "
                f"{median_rate:.1f} Hz"
            )

    # ── 4. Baseline-specific: motion contamination check ─────────────────────
    if class_name == 'Baseline':
        for s in group_stats:
            vt_mean = s.get('variance_turb_mean', 0)
            vt_p95  = s.get('variance_turb_p95', 0)
            if vt_mean > BASELINE_MAX_VARIANCE_TURB_MEAN:
                warnings.append(
                    f"{s['filename']}: BASELINE motion contamination — "
                    f"variance_turb mean={vt_mean:.4f} "
                    f"(threshold={BASELINE_MAX_VARIANCE_TURB_MEAN}). "
                    f"Robot vacuum / appliance / person nearby?"
                )
            elif vt_p95 > BASELINE_MAX_VARIANCE_TURB_P95:
                warnings.append(
                    f"{s['filename']}: BASELINE spike detected — "
                    f"variance_turb p95={vt_p95:.4f} "
                    f"(threshold={BASELINE_MAX_VARIANCE_TURB_P95}). "
                    f"Intermittent motion during recording?"
                )

    # ── 5. Motion-specific: suspiciously quiet session ────────────────────────
    if class_name in ('Quadrant_1', 'Quadrant_2'):
        for s in group_stats:
            vt_mean = s.get('variance_turb_mean', 0)
            if vt_mean < MOTION_MIN_VARIANCE_TURB_MEAN:
                warnings.append(
                    f"{s['filename']}: MOTION session is suspiciously quiet — "
                    f"variance_turb mean={vt_mean:.4f} "
                    f"(threshold={MOTION_MIN_VARIANCE_TURB_MEAN}). "
                    f"Was movement actually recorded? Possible mislabel?"
                )

    # ── 6. Minimum sample count ───────────────────────────────────────────────
    for s in group_stats:
        if s['n_samples'] < MIN_SAMPLES_PER_SESSION:
            warnings.append(
                f"{s['filename']}: only {s['n_samples']} samples "
                f"(minimum={MIN_SAMPLES_PER_SESSION}). "
                f"Session may be too short for reliable windows."
            )

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Verify ESPectre training data before model training.')
    parser.add_argument('--data-dir', default='csv_data',
                        help='Path to csv_data directory (default: csv_data)')
    parser.add_argument('--no-colour', action='store_true',
                        help='Disable ANSI colour output')
    args = parser.parse_args()

    if args.no_colour:
        for name in ['RESET','BOLD','RED','YELLOW','GREEN','CYAN','DIM']:
            globals()[name] = ''

    base_path = Path(args.data_dir)
    if not base_path.exists():
        print(err(f"Data directory not found: {base_path.absolute()}"))
        sys.exit(1)

    print(head("ESPectre Training Data Verification"))
    print(f"  Data directory : {base_path.absolute()}")
    print(f"  Z-score threshold        : {Z_SCORE_THRESHOLD}")
    print(f"  SC fingerprint threshold : {SC_FINGERPRINT_DEVIATION_THRESHOLD*100:.0f}%")
    print(f"  Baseline max vt mean     : {BASELINE_MAX_VARIANCE_TURB_MEAN}")
    print(f"  Motion min vt mean       : {MOTION_MIN_VARIANCE_TURB_MEAN}")
    print(f"  Min samples/session      : {MIN_SAMPLES_PER_SESSION}")

    # ── Load all files ────────────────────────────────────────────────────────
    # group_data[class_name][door_state] = list of session stat dicts
    group_data     = defaultdict(lambda: defaultdict(list))
    load_errors    = []
    total_files    = 0
    total_samples  = 0

    for dir_name, class_name in DIRECTORY_MAPPINGS.items():
        dir_path  = base_path / dir_name
        if not dir_path.exists():
            print(warn(f"Directory not found, skipping: {dir_path}"))
            continue

        csv_files = sorted(dir_path.glob('*.csv'))
        if not csv_files:
            print(warn(f"No CSV files in: {dir_path}"))
            continue

        for csv_file in csv_files:
            door_state = parse_door_state(csv_file.name)
            if door_state is None:
                load_errors.append(
                    f"{csv_file.name}: could not parse door state from filename")
                continue
            try:
                df    = pd.read_csv(csv_file)
                stats = compute_session_stats(df, csv_file.name)
                stats['class'] = class_name
                stats['door_state'] = door_state
                group_data[class_name][door_state].append(stats)
                total_files   += 1
                total_samples += stats['n_samples']
            except Exception as e:
                load_errors.append(f"{csv_file.name}: {e}")

    print(f"\n  Loaded {total_files} files / {total_samples:,} samples total")
    if load_errors:
        print(sub("Load Errors"))
        for e in load_errors:
            print(err(e))

    # ── Determine all known door states across the dataset ────────────────────
    all_door_states = sorted({
        ds
        for class_sessions in group_data.values()
        for ds in class_sessions.keys()
    })

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 1: Session count matrix
    # ═════════════════════════════════════════════════════════════════════════
    print(head("Session Count Matrix"))
    print(f"{'Door State':<14}", end='')
    for class_name in DIRECTORY_MAPPINGS.values():
        print(f"{class_name:>14}", end='')
    print()
    print('─' * (14 + 14 * len(DIRECTORY_MAPPINGS)))

    all_counts  = []
    missing_rows = []

    for ds in all_door_states:
        print(f"{ds:<14}", end='')
        row_counts = []
        for class_name in DIRECTORY_MAPPINGS.values():
            n = len(group_data[class_name].get(ds, []))
            row_counts.append(n)
            colour = GREEN if n >= 3 else (YELLOW if n > 0 else RED)
            print(c(f"{n:>14}", colour), end='')
        print()
        all_counts.append(row_counts)
        if any(c_ == 0 for c_ in row_counts):
            missing_rows.append(ds)

    print(f"\n  Colour key: {c('≥3 sessions', GREEN)}  "
          f"{c('1–2 sessions', YELLOW)}  "
          f"{c('0 sessions (missing)', RED)}")

    if missing_rows:
        print()
        for ds in missing_rows:
            for i, class_name in enumerate(DIRECTORY_MAPPINGS.values()):
                if len(group_data[class_name].get(ds, [])) == 0:
                    print(warn(f"Missing: {class_name} / door state {ds}"))

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 2: Per-group outlier detection
    # ═════════════════════════════════════════════════════════════════════════
    print(head("Outlier & Anomaly Detection"))

    total_warnings = 0
    all_warnings   = defaultdict(list)

    for class_name in DIRECTORY_MAPPINGS.values():
        class_warnings = 0
        print(sub(f"Class: {class_name}"))

        for ds in sorted(group_data[class_name].keys()):
            sessions = group_data[class_name][ds]
            n        = len(sessions)
            header   = f"  Door state {ds}  ({n} session{'s' if n != 1 else ''})"

            if n < MIN_SESSIONS_FOR_OUTLIER_DETECTION:
                print(f"{header}  {c('— too few sessions for outlier detection', DIM)}")
                continue

            warnings = check_group_outliers(sessions, class_name, ds)

            if warnings:
                print(c(header + f"  →  {len(warnings)} issue(s) found", YELLOW))
                for w in warnings:
                    print(err(f"     {w}"))
                class_warnings   += len(warnings)
                total_warnings   += len(warnings)
                all_warnings[class_name].extend(warnings)
            else:
                # Print compact stats line
                means = [f"{s['amp_mean_mean']:.2f}" if 'amp_mean_mean' in s
                         else f"{s.get('amp_mean_mean', '?')}"
                         for s in sessions]
                vt    = [f"{s['variance_turb_mean']:.3f}" for s in sessions]
                print(ok(
                    f"{header}  "
                    f"amp_mean=[{', '.join(means)}]  "
                    f"vt_mean=[{', '.join(vt)}]"
                ))

        if class_warnings == 0:
            print(ok(f"  No issues found for {class_name}"))

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3: Cross-class sanity checks (same door state)
    # ═════════════════════════════════════════════════════════════════════════
    print(head("Cross-Class Sanity Checks"))
    cross_issues = 0

    for ds in all_door_states:
        baseline_sessions = group_data['Baseline'].get(ds, [])
        q1_sessions       = group_data['Quadrant_1'].get(ds, [])
        q2_sessions       = group_data['Quadrant_2'].get(ds, [])

        if not baseline_sessions:
            continue

        baseline_vt = np.mean([s['variance_turb_mean'] for s in baseline_sessions])

        # Baseline should have lower variance_turb than motion classes
        for motion_class, motion_sessions in [('Quadrant_1', q1_sessions),
                                               ('Quadrant_2', q2_sessions)]:
            if not motion_sessions:
                continue
            motion_vt = np.mean([s['variance_turb_mean'] for s in motion_sessions])
            if baseline_vt >= motion_vt:
                print(err(
                    f"Door state {ds}: Baseline vt_mean ({baseline_vt:.4f}) ≥ "
                    f"{motion_class} vt_mean ({motion_vt:.4f}) — "
                    f"motion class should have higher turbulence"
                ))
                cross_issues += 1

    if cross_issues == 0:
        print(ok("All door states: Baseline turbulence < Motion turbulence ✓"))

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 4: Per-session detail table
    # ═════════════════════════════════════════════════════════════════════════
    print(head("Per-Session Detail Table"))

    col_w = [40, 7, 8, 7, 9, 9, 9]
    header_row = (
        f"{'Filename':<{col_w[0]}} "
        f"{'Samples':>{col_w[1]}} "
        f"{'Rate(Hz)':>{col_w[2]}} "
        f"{'DS':>{col_w[3]}} "
        f"{'amp_mean':>{col_w[4]}} "
        f"{'vt_mean':>{col_w[5]}} "
        f"{'vt_p95':>{col_w[6]}}"
    )
    print(c(header_row, BOLD))
    print('─' * sum(col_w + [len(col_w)]))

    for class_name in DIRECTORY_MAPPINGS.values():
        print(c(f"\n  {class_name}", CYAN))
        for ds in sorted(group_data[class_name].keys()):
            for s in group_data[class_name][ds]:
                fname   = s['filename']
                # Truncate filename to fit column
                if len(fname) > col_w[0]:
                    fname = '…' + fname[-(col_w[0]-1):]

                is_flagged = any(
                    s['filename'] in w
                    for w in all_warnings[class_name]
                )
                colour = RED if is_flagged else RESET

                line = (
                    f"{fname:<{col_w[0]}} "
                    f"{s['n_samples']:>{col_w[1]}} "
                    f"{s['rate_hz']:>{col_w[2]}.1f} "
                    f"{s['door_state']:>{col_w[3]}} "
                    f"{s.get('amp_mean_mean', 0):>{col_w[4]}.4f} "
                    f"{s.get('variance_turb_mean', 0):>{col_w[5]}.4f} "
                    f"{s.get('variance_turb_p95', 0):>{col_w[6]}.4f}"
                )
                print(c(f"  {line}", colour) if is_flagged else f"  {line}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5: Summary
    # ═════════════════════════════════════════════════════════════════════════
    print(head("Summary"))
    print(f"  Total files loaded   : {total_files}")
    print(f"  Total samples        : {total_samples:,}")
    print(f"  Classes found        : {list(group_data.keys())}")
    print(f"  Door states found    : {all_door_states}")
    print(f"  Load errors          : {len(load_errors)}")
    print(f"  Cross-class issues   : {cross_issues}")
    print(f"  Outlier warnings     : {total_warnings}")

    if total_warnings == 0 and cross_issues == 0 and len(load_errors) == 0:
        print(c(f"\n  ✓ Dataset looks clean — safe to train", GREEN + BOLD))
    else:
        print(c(f"\n  ⚠ Review flagged sessions before training", YELLOW + BOLD))
        if total_warnings > 0:
            print(c(f"\n  Flagged sessions (review or re-record):", YELLOW))
            seen = set()
            for warnings in all_warnings.values():
                for w in warnings:
                    fname = w.split(':')[0].strip()
                    if fname not in seen:
                        print(f"    • {fname}")
                        seen.add(fname)

    print()


if __name__ == '__main__':
    main()
