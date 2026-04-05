import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, cross_val_predict
from scipy.stats import randint
from pathlib import Path
import joblib


NULL_SUBCARRIER_INDICES = {0, 1, 2, 3, 4, 5, 27, 28, 29, 30, 31, 32, 33, 59, 60, 61, 62, 63}

# Normalisation strategy per aggregate feature index:
#   LINEAR  (÷ sc_global_mean)  : entropy_turb(0), iqr_turb(1), amp family(5-10)
#   QUADRATIC (÷ sc_global_mean²): variance_turb(2)
#   NONE (raw)                  : skewness(3), kurtosis(4), phase_diff_*(11-14)
LINEAR_NORM_INDICES    = {0, 1, 5, 6, 7, 8, 9, 10}
QUADRATIC_NORM_INDICES = {2}


def get_sc_feature_columns(dataframe):
    all_sc_cols = sorted(
        [c for c in dataframe.columns if c.startswith('sc_amp_')],
        key=lambda c: int(c.split('_')[-1])
    )
    if not all_sc_cols:
        return [], []

    dropped, valid = [], []
    for col in all_sc_cols:
        idx = int(col.split('_')[-1])
        if idx in NULL_SUBCARRIER_INDICES:
            dropped.append((col, 'known null subcarrier (802.11n HT20)'))
            continue
        if dataframe[col].std() == 0:
            dropped.append((col, f'zero variance (all values = {dataframe[col].iloc[0]:.1f})'))
            continue
        valid.append(col)

    return valid, dropped


def build_windowed_features(dataframe, aggregate_features, valid_sc_cols,
                             seq_len=28, stride=8):
    """
    Build flat feature vectors from sliding windows over each session.

    Normalisation is per-window so features are invariant to session-level
    AGC drift, temperature shift, and time-of-day amplitude variation:

      - Amplitude-linear aggregates (entropy, iqr, amp family):
            ÷ sc_global_mean

      - variance_turb:
            ÷ sc_global_mean²  (variance scales with amplitude²)

      - skewness, kurtosis, phase_diff_*:
            kept raw (dimensionless)

      - Per-subcarrier amplitudes:
            ÷ sc_global_mean  (preserves relative spatial fingerprint)

    Returns X, y, and group IDs (one per source file) for GroupKFold.
    """
    X_rows, y_rows, group_rows = [], [], []
    group_id = 0

    for source_file, group in dataframe.groupby('source_file'):
        group   = group.reset_index(drop=True)
        n       = len(group)
        sc_arr  = group[valid_sc_cols].values.astype('float32')
        agg_arr = group[aggregate_features].values.astype('float32')
        label   = group['label'].iloc[0]

        for start in range(0, n - seq_len + 1, stride):
            end     = start + seq_len
            sc_win  = sc_arr[start:end]
            agg_win = agg_arr[start:end]

            sc_global_mean  = float(np.mean(sc_win)) + 1e-6
            sc_global_mean2 = sc_global_mean ** 2

            feats = []

            # 1. Aggregate feature stats — 15 × 5 = 75
            for i in range(agg_win.shape[1]):
                col = agg_win[:, i].copy()
                if i in QUADRATIC_NORM_INDICES:
                    col = col / sc_global_mean2
                elif i in LINEAR_NORM_INDICES:
                    col = col / sc_global_mean
                feats += [
                    float(np.mean(col)),
                    float(np.std(col)),
                    float(np.max(col)),
                    float(np.percentile(col, 25)),
                    float(np.percentile(col, 75)),
                ]

            # 2. Per-subcarrier normalised mean + std — 31 × 2 = 62
            sc_norm = sc_win / sc_global_mean
            feats.extend(np.mean(sc_norm, axis=0).tolist())
            feats.extend(np.std(sc_norm,  axis=0).tolist())

            # 3. Spike features on variance_turb — 4
            vt      = agg_win[:, 2] / sc_global_mean2
            vt_mean = float(np.mean(vt))
            vt_max  = float(np.max(vt))
            vt_std  = float(np.std(vt))
            feats += [
                vt_max,
                vt_std,
                float(np.sum(vt > vt_mean + 2 * vt_std)),
                float(vt_max / (vt_mean + 1e-9)),
            ]

            # 4. Temporal trend — 1
            half = seq_len // 2
            feats.append(float(np.mean(vt[half:]) - np.mean(vt[:half])))

            # Total: 75 + 62 + 4 + 1 = 142
            X_rows.append(feats)
            y_rows.append(label)
            group_rows.append(group_id)

        group_id += 1

    return (
        np.array(X_rows,     dtype='float32'),
        np.array(y_rows),
        np.array(group_rows, dtype='int32'),
    )


def load_all_data(base_dir='csv_data'):
    """
    Load all CSV files from all class directories and build windowed features
    across the full dataset. No train/test split — all data is used for training.

    GroupKFold CV inside the HPO search provides honest generalisation estimates
    by ensuring each fold contains only unseen recording sessions.
    """
    base_path = Path(base_dir)

    directory_mappings = {
        'baseline':    'Baseline',
        'movement_q1': 'Quadrant_1',
        'movement_q2': 'Quadrant_2',
    }

    aggregate_features = [
        'entropy_turb', 'iqr_turb', 'variance_turb',       # indices 0-2
        'skewness', 'kurtosis',                              # indices 3-4
        'amp_mean', 'amp_range', 'amp_std',                  # indices 5-7
        'amp_mean_low', 'amp_mean_mid', 'amp_mean_high',     # indices 8-10
        'phase_diff_mean', 'phase_diff_std',                 # indices 11-12
        'phase_diff_range', 'phase_diff_skew',               # indices 13-14
    ]

    print(f"{'='*80}")
    print(f"Loading data from: {base_path.absolute()}")
    print(f"{'='*80}\n")

    all_dfs = []

    for dir_name, label_name in directory_mappings.items():
        dir_path  = base_path / dir_name
        if not dir_path.exists():
            print(f"⚠️  Warning: Directory '{dir_path}' not found, skipping...")
            continue

        csv_files = sorted(dir_path.glob('*.csv'))
        if not csv_files:
            print(f"⚠️  Warning: No CSV files found in '{dir_path}', skipping...")
            continue

        print(f"📁 {dir_name}/ → Label: '{label_name}'")
        print(f"   Found {len(csv_files)} CSV file(s)")

        total_samples = 0
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['label']       = label_name
                df['source_file'] = csv_file.name
                all_dfs.append(df)
                total_samples += len(df)
                print(f"      ✓ {csv_file.name}: {len(df)} samples")
            except Exception as e:
                print(f"      ✗ Error loading {csv_file.name}: {e}")

        print(f"   Total: {total_samples} samples for '{label_name}'\n")

    if not all_dfs:
        raise ValueError("No data loaded! Check your directory structure and CSV files.")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    print(f"{'='*80}")
    print(f"COMBINED DATASET")
    print(f"{'='*80}")
    print(f"Total samples loaded: {len(combined_df)}")
    print(f"\nClass distribution:")
    for lname, count in combined_df['label'].value_counts().items():
        print(f"  {lname:15s}: {count:5d} samples ({count/len(combined_df)*100:5.1f}%)")

    missing = [f for f in aggregate_features if f not in combined_df.columns]
    if missing:
        raise ValueError(f"Missing aggregate features in CSV files: {missing}")

    valid_sc_cols, dropped_sc_cols = get_sc_feature_columns(combined_df)

    if dropped_sc_cols:
        print(f"\n{'='*80}")
        print("SUBCARRIER FILTERING")
        print(f"{'='*80}")
        print(f"Dropped {len(dropped_sc_cols)} uninformative subcarrier column(s):")
        for col, reason in dropped_sc_cols:
            print(f"  ✗ {col:12s}  ({reason})")
        if valid_sc_cols:
            print(f"\nRetained {len(valid_sc_cols)} informative subcarrier column(s):")
            indices = [int(c.split('_')[-1]) for c in valid_sc_cols]
            print(f"  ✓ Indices: {indices}")

    print(f"\n{'='*80}")
    print("FEATURE SET SUMMARY")
    print(f"{'='*80}")
    print(f"  Aggregate features  : {len(aggregate_features)}")
    print(f"  Subcarrier features : {len(valid_sc_cols)}")
    print(f"  Total features      : {len(aggregate_features) + len(valid_sc_cols)}")

    unique_labels = sorted(combined_df['label'].unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    print(f"\n{'='*80}")
    print("BUILDING WINDOWS (full dataset)")
    print(f"{'='*80}")
    X_all, y_all_labels, groups_all = build_windowed_features(
        combined_df, aggregate_features, valid_sc_cols, seq_len=28, stride=8)
    y_all = np.array([label_mapping[l] for l in y_all_labels])

    idx_to_label = {v: k for k, v in label_mapping.items()}

    print(f"  Total windows    : {len(y_all)}")
    print(f"  Features/window  : {X_all.shape[1]}  (expected 142)")
    print(f"\n  Class distribution:")
    for idx in sorted(label_mapping.values()):
        count = np.sum(y_all == idx)
        print(f"    {idx_to_label[idx]:15s}: {count:5d} windows ({count/len(y_all)*100:.1f}%)")

    print(f"\n  Unique recording sessions (CV groups): {len(np.unique(groups_all))}")

    window_feature_names = (
        [f"{f}_{s}" for f in aggregate_features for s in ['mean', 'std', 'max', 'p25', 'p75']]
        + [f"{c}_wmean" for c in valid_sc_cols]
        + [f"{c}_wstd"  for c in valid_sc_cols]
        + ['vt_max', 'vt_std', 'vt_spike_count', 'vt_snr', 'vt_trend']
    )

    return X_all, y_all, groups_all, window_feature_names, label_mapping


def train_random_forest(X_all, y_all, groups_all, feature_names, label_mapping):
    """
    Train a 3-class Random Forest on the full dataset.

    No train/test split — all data is used for training so the model sees
    the full range of time-of-day, door states, and sessions.

    Generalisation is estimated via session-aware GroupKFold CV inside the
    HPO search (each CV fold holds out complete recording sessions, never
    individual windows from a seen session).

    No StandardScaler — RF is scale-invariant. All amplitude invariance is
    handled in build_windowed_features via per-window sc_global_mean /
    sc_global_mean² normalisation.
    """
    idx_to_label = {idx: label for label, idx in label_mapping.items()}
    class_names  = [idx_to_label[i] for i in sorted(label_mapping.values())]

    print(f"{'='*80}")
    print("HYPERPARAMETER SEARCH (RandomizedSearchCV + GroupKFold)")
    print(f"{'='*80}")
    print(f"  Total windows     : {len(y_all)}")
    print(f"  Search iterations : 30")
    print(f"  CV folds          : 5 (session-aware GroupKFold)")
    print(f"  Scoring           : balanced_accuracy")
    print(f"  n_jobs            : -1 (all cores)\n")

    param_dist = {
        'n_estimators':      randint(100, 350),
        'max_depth':         [8, 10, 12, 15, 20],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf':  randint(2, 10),
        'max_features':      ['sqrt', 'log2'],
    }

    search = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_distributions=param_dist,
        n_iter=30,
        cv=GroupKFold(n_splits=5),
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    search.fit(X_all, y_all, groups=groups_all)
    best_model = search.best_estimator_

    print(f"\n  Best CV balanced_accuracy : {search.best_score_:.4f}")
    print(f"  Best parameters:")
    for k, v in sorted(search.best_params_.items()):
        print(f"    {k:22s}: {v}")

    # ------------------------------------------------------------------ #
    #  CV PREDICTIONS — honest per-class performance estimate             #
    #  Uses the same GroupKFold so each window is predicted by a model    #
    #  that never saw its source session during training.                 #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*80}")
    print("SESSION-AWARE CV PERFORMANCE ESTIMATE")
    print(f"{'='*80}")
    print("(Each window predicted by a fold that excluded its source session)\n")

    y_cv_pred = cross_val_predict(
        best_model,
        X_all, y_all,
        cv=GroupKFold(n_splits=5),
        groups=groups_all,
        n_jobs=-1,
    )

    cv_accuracy = accuracy_score(y_all, y_cv_pred)
    print(f"CV Accuracy : {cv_accuracy:.4f} ({cv_accuracy*100:.2f}%)\n")
    print(classification_report(y_all, y_cv_pred, target_names=class_names, digits=4))

    print(f"{'='*80}")
    print("CONFUSION MATRIX (CV predictions)")
    print(f"{'='*80}")
    conf = confusion_matrix(y_all, y_cv_pred)
    print(f"\n{'':20s}Predicted:")
    print(f"{'Actual:':20s}", end='')
    for cn in class_names:
        print(f"{cn:>15s}", end='')
    print()
    for i, cn in enumerate(class_names):
        print(f"  {cn:18s}", end='')
        for j in range(len(class_names)):
            print(f"{conf[i][j]:15d}", end='')
        print()

    # ------------------------------------------------------------------ #
    #  FINAL MODEL — retrained on ALL data with best hyperparameters      #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*80}")
    print("FINAL MODEL — retraining on full dataset with best parameters")
    print(f"{'='*80}\n")

    best_model.fit(X_all, y_all)
    train_accuracy = accuracy_score(y_all, best_model.predict(X_all))
    print(f"  Full-data training accuracy : {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  (Expected to be high — model has seen all data)")

    # Feature importances
    importance_indices = np.argsort(best_model.feature_importances_)[::-1]
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCES (Ranked)")
    print(f"{'='*80}")
    print(f"{'Rank':>4s} {'Feature':25s} {'Importance':>12s} {'Percentage':>12s}")
    print(f"{'-'*80}")
    for i, fi in enumerate(importance_indices):
        imp = best_model.feature_importances_[fi]
        pct = imp * 100
        bar = '█' * int(pct * 0.5)
        print(f"{i+1:3d}. {feature_names[fi]:25s} {imp:11.4f} ({pct:5.1f}%) {bar}")
    print(f"{'='*80}\n")

    return best_model, label_mapping


def main():
    print(f"\n{'='*80}")
    print("ESPectre WiFi Sensing - Random Forest Classifier Training")
    print(f"{'='*80}\n")

    X_all, y_all, groups_all, feature_names, label_mapping = \
        load_all_data('csv_data')

    trained_model, label_map = train_random_forest(
        X_all, y_all, groups_all, feature_names, label_mapping)

    model_filename = 'rf_spatial_classifier.pkl'
    joblib.dump(trained_model, model_filename)
    joblib.dump(label_map,     'rf_label_mapping.pkl')
    joblib.dump(feature_names, 'rf_feature_names.pkl')

    print(f"{'='*80}")
    print(f"✓ MODEL SAVED")
    print(f"{'='*80}")
    print(f"  Filename  : {model_filename}")
    print(f"  Size      : {Path(model_filename).stat().st_size / 1024:.1f} KB")
    print(f"  Classes   : {list(label_map.keys())}")
    print(f"  Features  : {len(feature_names)}")
    print(f"\n  Also saved:")
    print(f"    rf_label_mapping.pkl  — class index ↔ name mapping")
    print(f"    rf_feature_names.pkl  — feature name list (for debugging)")
    print(f"{'='*80}\n")

    return trained_model, label_map


if __name__ == '__main__':
    model, label_mapping = main()
