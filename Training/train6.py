import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib


NULL_SUBCARRIER_INDICES = {0, 1, 2, 3, 4, 5, 27, 28, 29, 30, 31, 32, 33, 59, 60, 61, 62, 63}


def get_sc_feature_columns(dataframe):
    all_sc_cols = sorted(
        [c for c in dataframe.columns if c.startswith('sc_amp_')],
        key=lambda c: int(c.split('_')[-1])
    )

    if not all_sc_cols:
        return [], []

    dropped = []
    valid = []

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
    sc_cols = valid_sc_cols
    X_rows, y_rows = [], []

    for source_file, group in dataframe.groupby('source_file'):
        group   = group.reset_index(drop=True)
        n       = len(group)
        sc_arr  = group[sc_cols].values.astype('float32')
        agg_arr = group[aggregate_features].values.astype('float32')
        label   = group['label'].iloc[0]

        for start in range(0, n - seq_len + 1, stride):
            end = start + seq_len
            sc_win  = sc_arr[start:end]
            agg_win = agg_arr[start:end]

            feats = []

            # 1. Per-aggregate-feature: mean, std, max, p25, p75 — 15 × 5 = 75
            for i in range(agg_win.shape[1]):
                col = agg_win[:, i]
                feats += [
                    float(np.mean(col)),
                    float(np.std(col)),
                    float(np.max(col)),
                    float(np.percentile(col, 25)),
                    float(np.percentile(col, 75)),
                ]

            # 2. Per-subcarrier mean + std — 31 × 2 = 62
            feats.extend(np.mean(sc_win, axis=0).tolist())
            feats.extend(np.std(sc_win,  axis=0).tolist())

            # 3. Spike features on variance_turb (index 2) — 4
            vt      = agg_win[:, 2]
            vt_mean = float(np.mean(vt))
            vt_max  = float(np.max(vt))
            vt_std  = float(np.std(vt))
            feats += [
                vt_max,
                vt_std,
                float(np.sum(vt > vt_mean + 2*vt_std)),
                float(vt_max / (vt_mean + 1e-6)),
            ]

            # 4. Temporal trend — 1
            half = seq_len // 2
            feats.append(float(np.mean(vt[half:]) - np.mean(vt[:half])))

            # Total: 75 + 62 + 4 + 1 = 142
            X_rows.append(feats)
            y_rows.append(label)

    return np.array(X_rows, dtype='float32'), np.array(y_rows)


def load_data_from_directories(base_dir='csv_data'):
    base_path = Path(base_dir)

    directory_mappings = {
        'baseline':    'Baseline',
        'movement_q1': 'Quadrant_1',
        'movement_q2': 'Quadrant_2',
    }

    aggregate_features = [
        'entropy_turb', 'iqr_turb', 'variance_turb',
        'skewness', 'kurtosis',
        'amp_mean', 'amp_range', 'amp_std',
        'amp_mean_low', 'amp_mean_mid', 'amp_mean_high',
        'phase_diff_mean', 'phase_diff_std',
        'phase_diff_range', 'phase_diff_skew',
    ]

    print(f"{'='*80}")
    print(f"Loading data from: {base_path.absolute()}")
    print(f"{'='*80}\n")

    all_dfs = []

    for dir_name, label_name in directory_mappings.items():
        dir_path = base_path / dir_name

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

    # ── Subcarrier filtering ─────────────────────────────────────────────
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

    # ── Random 80/20 split ───────────────────────────────────────────────
    unique_labels = sorted(combined_df['label'].unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    print(f"\n{'='*80}")
    print("TRAIN/TEST SPLIT (random 80/20, stratified by label)")
    print(f"{'='*80}")

    train_df, test_df = train_test_split(
        combined_df,
        test_size=0.2,
        random_state=42,
        stratify=combined_df['label']
    )

    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples:  {len(test_df)}")

    # ── Build windowed features ──────────────────────────────────────────
    print(f"\n  Building training windows...")
    X_train, y_train_labels = build_windowed_features(
        train_df, aggregate_features, valid_sc_cols, seq_len=28, stride=8)
    y_train = np.array([label_mapping[l] for l in y_train_labels])

    print(f"  Building test windows...")
    X_test, y_test_labels = build_windowed_features(
        test_df, aggregate_features, valid_sc_cols, seq_len=28, stride=8)
    y_test = np.array([label_mapping[l] for l in y_test_labels])

    print(f"\n{'='*80}")
    print("WINDOW SUMMARY")
    print(f"{'='*80}")
    print(f"  Training windows : {len(y_train)}")
    print(f"  Test windows     : {len(y_test)}")
    print(f"  Features/window  : {X_train.shape[1]}")

    print(f"\n  Training class distribution:")
    idx_to_label = {v: k for k, v in label_mapping.items()}
    for idx in sorted(label_mapping.values()):
        count = np.sum(y_train == idx)
        print(f"    {idx_to_label[idx]:15s}: {count:4d} windows")

    print(f"\n  Test class distribution:")
    for idx in sorted(label_mapping.values()):
        count = np.sum(y_test == idx)
        print(f"    {idx_to_label[idx]:15s}: {count:4d} windows")

    window_feature_names = (
        [f"{f}_{s}" for f in aggregate_features
                    for s in ['mean', 'std', 'max', 'p25', 'p75']]   # 15 × 5 = 75
        + [f"{c}_wmean" for c in valid_sc_cols]                       # 31
        + [f"{c}_wstd"  for c in valid_sc_cols]                       # 31
        + ['vt_max', 'vt_std', 'vt_spike_count', 'vt_snr', 'vt_trend']  # 5
    )
    # Total: 75 + 62 + 5 = 142

    return X_train, X_test, y_train, y_test, window_feature_names, label_mapping


def train_random_forest(X_train, X_test, y_train, y_test, feature_names, label_mapping):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"{'='*80}")
    print(f"TRAIN/TEST SPLIT")
    print(f"{'='*80}")
    total = len(X_train) + len(X_test)
    print(f"Training set size: {len(X_train)} samples ({len(X_train)/total*100:.1f}%)")
    print(f"Test set size:     {len(X_test)} samples ({len(X_test)/total*100:.1f}%)")

    idx_to_label = {idx: label for label, idx in label_mapping.items()}

    random_forest_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )

    print(f"\n{'='*80}")
    print("TRAINING RANDOM FOREST")
    print(f"{'='*80}")
    print(f"Parameters:")
    print(f"  - Estimators       : {random_forest_classifier.n_estimators}")
    print(f"  - Max depth        : {random_forest_classifier.max_depth}")
    print(f"  - Min samples split: {random_forest_classifier.min_samples_split}")
    print(f"  - Features per tree: {random_forest_classifier.max_features}")
    print(f"  - Total features   : {len(feature_names)}")
    print()

    random_forest_classifier.fit(X_train, y_train)

    y_pred_train = random_forest_classifier.predict(X_train)
    y_pred_test  = random_forest_classifier.predict(X_test)

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
    confusion = confusion_matrix(y_test, y_pred_test)

    print(f"\n{'':20s}Predicted:")
    print(f"{'Actual:':20s}", end='')
    for cn in class_names:
        print(f"{cn:>15s}", end='')
    print()

    for i, cn in enumerate(class_names):
        print(f"  {cn:18s}", end='')
        for j in range(len(class_names)):
            print(f"{confusion[i][j]:15d}", end='')
        print()

    feature_importances = random_forest_classifier.feature_importances_
    importance_indices  = np.argsort(feature_importances)[::-1]

    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCES (Ranked)")
    print(f"{'='*80}")
    print(f"{'Rank':>4s} {'Feature':25s} {'Importance':>12s} {'Percentage':>12s}")
    print(f"{'-'*80}")

    for i in range(len(feature_names)):
        fi   = importance_indices[i]
        name = feature_names[fi]
        imp  = feature_importances[fi]
        pct  = imp * 100
        bar  = '█' * int(pct * 0.5)
        print(f"{i+1:3d}. {name:25s} {imp:11.4f} ({pct:5.1f}%) {bar}")

    print(f"{'='*80}\n")

    return random_forest_classifier, label_mapping, scaler


def main():
    print(f"\n{'='*80}")
    print("ESPectre WiFi Sensing - Random Forest Classifier Training")
    print(f"{'='*80}\n")

    X_train, X_test, y_train, y_test, feature_names, label_mapping = \
        load_data_from_directories('csv_data')

    trained_model, label_map, scaler = train_random_forest(
        X_train, X_test, y_train, y_test, feature_names, label_mapping)

    model_filename = 'rf_spatial_classifier.pkl'
    joblib.dump(trained_model, model_filename)
    joblib.dump(scaler, 'rf_scaler.pkl')
    print("✓ Scaler saved: rf_scaler.pkl")

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
