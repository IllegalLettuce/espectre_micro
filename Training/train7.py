import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from scipy.stats import randint
from pathlib import Path
import joblib


NULL_SUBCARRIER_INDICES = {0, 1, 2, 3, 4, 5, 27, 28, 29, 30, 31, 32, 33, 59, 60, 61, 62, 63}

# VT_MEAN_IDX: variance_turb is aggregate index 2, mean is the 0th stat → 2*5+0 = 10
VT_MEAN_IDX = 10

# Normalisation strategy per aggregate feature index:
#   LINEAR  (÷ sc_global_mean)  : iqr_turb, entropy_turb, amp_mean family
#                                  These are amplitude-linear measures.
#   QUADRATIC (÷ sc_global_mean²): variance_turb only.
#                                  Variance of raw amplitudes scales with amp²,
#                                  so a single linear division still leaves a
#                                  residual amplitude dependency.
#   NONE                         : skewness, kurtosis, phase_diff_*
#                                  These are already dimensionless or relative.
LINEAR_NORM_INDICES    = {0, 1, 5, 6, 7, 8, 9, 10}  # entropy, iqr, amp family
QUADRATIC_NORM_INDICES = {2}                           # variance_turb


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

    Normalisation per aggregate feature type:
      - entropy_turb, iqr_turb, amp_mean family: divided by sc_global_mean
        (amplitude-linear measures)
      - variance_turb: divided by sc_global_mean² (scales with amplitude²)
      - skewness, kurtosis, phase_diff_*: kept raw (dimensionless)
      - Per-subcarrier amplitudes: divided by sc_global_mean

    All normalisation is window-local (per-window sc_global_mean), making
    features invariant to session-level AGC/temperature amplitude drift.

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
            #    Divide by sc_global_mean² for the same reason as section 1
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


def load_data_from_directories(base_dir='csv_data'):
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

    all_dfs        = []
    files_by_class = {}

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

        files_by_class[label_name] = []
        total_samples = 0

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['label']       = label_name
                df['source_file'] = csv_file.name
                all_dfs.append(df)
                files_by_class[label_name].append((csv_file, df))
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

    train_dfs, test_dfs = [], []

    print(f"\n{'='*80}")
    print("SESSION-LEVEL TRAIN/TEST SPLIT")
    print(f"{'='*80}")

    for label_name, file_df_pairs in files_by_class.items():
        door_state_groups = {}
        for csv_file, df in file_df_pairs:
            parts      = csv_file.stem.split('_')
            door_state = '_'.join(parts[-5:-2])
            door_state_groups.setdefault(door_state, []).append((csv_file, df))

        print(f"\n  {label_name}:")

        for door_state, pairs in sorted(door_state_groups.items()):
            if len(pairs) < 2:
                print(f"    ⚠️  {door_state}: only {len(pairs)} session — training only")
                for _, df in pairs:
                    train_dfs.append(df)
                continue

            test_csv, test_df = pairs[-1]
            train_pairs       = pairs[:-1]
            print(f"    {door_state} → test: {test_csv.name} ({len(test_df)} samples), "
                  f"train: {len(train_pairs)} session(s)")
            test_dfs.append(test_df)
            for _, df in train_pairs:
                train_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df  = pd.concat(test_dfs,  ignore_index=True)

    print(f"\n  Building training windows...")
    X_train, y_train_labels, groups_train = build_windowed_features(
        train_df, aggregate_features, valid_sc_cols, seq_len=28, stride=8)
    y_train = np.array([label_mapping[l] for l in y_train_labels])

    print(f"  Building test windows...")
    X_test, y_test_labels, _ = build_windowed_features(
        test_df, aggregate_features, valid_sc_cols, seq_len=28, stride=8)
    y_test = np.array([label_mapping[l] for l in y_test_labels])

    print(f"\n{'='*80}")
    print("WINDOW SUMMARY")
    print(f"{'='*80}")
    print(f"  Training windows : {len(y_train)}")
    print(f"  Test windows     : {len(y_test)}")
    print(f"  Features/window  : {X_train.shape[1]}  (expected 142)")

    idx_to_label = {v: k for k, v in label_mapping.items()}
    print(f"\n  Training class distribution:")
    for idx in sorted(label_mapping.values()):
        print(f"    {idx_to_label[idx]:15s}: {np.sum(y_train == idx):4d} windows")
    print(f"\n  Test class distribution:")
    for idx in sorted(label_mapping.values()):
        print(f"    {idx_to_label[idx]:15s}: {np.sum(y_test == idx):4d} windows")

    # Diagnostic: verify vt normalisation achieved train/test alignment
    print(f"\n{'='*80}")
    print("VARIANCE_TURB_MEAN DIAGNOSTIC (÷ sc_global_mean²)")
    print(f"{'='*80}")
    print(f"  {'Class':12s}  {'Split':6s}  {'mean':9s}  {'p10':9s}  {'p90':9s}  {'max':9s}")
    for split_name, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        for idx in sorted(label_mapping.values()):
            mask = y == idx
            vals = X[mask, VT_MEAN_IDX]
            print(f"  {idx_to_label[idx]:12s}  {split_name:6s}  "
                  f"{np.mean(vals):9.6f}  {np.percentile(vals,10):9.6f}  "
                  f"{np.percentile(vals,90):9.6f}  {np.max(vals):9.6f}")

    window_feature_names = (
        [f"{f}_{s}" for f in aggregate_features for s in ['mean', 'std', 'max', 'p25', 'p75']]
        + [f"{c}_wmean" for c in valid_sc_cols]
        + [f"{c}_wstd"  for c in valid_sc_cols]
        + ['vt_max', 'vt_std', 'vt_spike_count', 'vt_snr', 'vt_trend']
    )

    return X_train, X_test, y_train, y_test, window_feature_names, label_mapping, groups_train


def train_random_forest(X_train, X_test, y_train, y_test,
                        feature_names, label_mapping, groups_train):
    """
    Single 3-class Random Forest classifier.

    No StandardScaler — RF is scale-invariant (split decisions use only
    feature ordering). Scaling was causing distribution shift because the
    scaler was fit on morning training sessions and applied to afternoon
    test sessions. All amplitude invariance is handled in build_windowed_features
    via per-window sc_global_mean / sc_global_mean² normalisation.

    The two-stage threshold approach was abandoned because Q1 (hallway motion)
    is weak-motion that genuinely overlaps with Baseline in variance_turb space.
    The RF uses all 142 features jointly (subcarrier spatial fingerprint +
    turbulence statistics) to make this distinction where a single threshold
    on one feature cannot.
    """
    idx_to_label = {idx: label for label, idx in label_mapping.items()}

    total = len(X_train) + len(X_test)
    print(f"{'='*80}")
    print(f"TRAIN/TEST SPLIT")
    print(f"{'='*80}")
    print(f"Training set size: {len(X_train)} samples ({len(X_train)/total*100:.1f}%)")
    print(f"Test set size:     {len(X_test)} samples ({len(X_test)/total*100:.1f}%)")

    print(f"\n{'='*80}")
    print("HYPERPARAMETER SEARCH (RandomizedSearchCV + GroupKFold)")
    print(f"{'='*80}")
    print(f"  Search iterations : 50")
    print(f"  CV folds          : 5 (session-aware)")
    print(f"  Scoring           : balanced_accuracy")
    print(f"  n_jobs            : -1 (all cores)\n")

    param_dist = {
        'n_estimators':      randint(100, 600),
        'max_depth':         [8, 10, 12, 15, 20, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf':  randint(1, 10),
        'max_features':      ['sqrt', 'log2', 0.3, 0.5],
    }

    search = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_distributions=param_dist,
        n_iter=50,
        cv=GroupKFold(n_splits=5),
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    search.fit(X_train, y_train, groups=groups_train)
    best_model = search.best_estimator_

    print(f"\n  Best CV balanced_accuracy : {search.best_score_:.4f}")
    print(f"  Best parameters:")
    for k, v in sorted(search.best_params_.items()):
        print(f"    {k:22s}: {v}")

    y_pred_train = best_model.predict(X_train)
    y_pred_test  = best_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy  = accuracy_score(y_test,  y_pred_test)

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    gap = train_accuracy - test_accuracy
    if gap > 0.1:
        print(f"⚠️  Warning: Possible overfitting (train-test gap: {gap*100:.1f}%)")
    else:
        print(f"✓ Good generalization (train-test gap: {gap*100:.1f}%)")

    class_names = [idx_to_label[i] for i in sorted(label_mapping.values())]

    print(f"\n{'='*80}")
    print("CLASSIFICATION REPORT (Test Set)")
    print(f"{'='*80}")
    print(classification_report(y_test, y_pred_test, target_names=class_names, digits=4))

    print(f"{'='*80}")
    print("CONFUSION MATRIX (Test Set)")
    print(f"{'='*80}")
    conf = confusion_matrix(y_test, y_pred_test)
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

    X_train, X_test, y_train, y_test, feature_names, label_mapping, groups_train = \
        load_data_from_directories('csv_data')

    trained_model, label_map = train_random_forest(
        X_train, X_test, y_train, y_test, feature_names, label_mapping, groups_train)

    model_filename = 'rf_spatial_classifier.pkl'
    joblib.dump(trained_model, model_filename)
    joblib.dump(label_map,     'rf_label_mapping.pkl')

    print(f"{'='*80}")
    print(f"✓ MODEL SAVED")
    print(f"{'='*80}")
    print(f"Filename : {model_filename}")
    print(f"Size     : {Path(model_filename).stat().st_size / 1024:.1f} KB")
    print(f"Classes  : {list(label_map.keys())}")
    print(f"Features : {len(feature_names)}")
    print(f"{'='*80}\n")

    return trained_model, label_map


if __name__ == '__main__':
    model, label_mapping = main()
