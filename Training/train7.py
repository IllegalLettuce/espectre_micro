import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from scipy.stats import randint
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
    # Indices of absolute-amplitude aggregate features that need normalisation
    # amp_mean=5, amp_range=6, amp_std=7, amp_mean_low=8, amp_mean_mid=9, amp_mean_high=10
    AMP_AGG_INDICES = {5, 6, 7, 8, 9, 10}

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

            # Reference amplitude for this window — used to normalise
            # absolute-amplitude features
            sc_global_mean = float(np.mean(sc_win)) + 1e-6

            feats = []

            # 1. Aggregate features — RAW stats (75)
            #    variance_turb, entropy_turb, iqr_turb, skewness, kurtosis,
            #    phase_diff_* are already relative: keep them raw.
            #    amp_mean family are absolute: divide by sc_global_mean.
            for i in range(agg_win.shape[1]):
                col = agg_win[:, i]
                if i in AMP_AGG_INDICES:
                    col = col / sc_global_mean   # remove session DC offset
                feats += [
                    float(np.mean(col)),
                    float(np.std(col)),
                    float(np.max(col)),
                    float(np.percentile(col, 25)),   # raw — preserves magnitude
                    float(np.percentile(col, 75)),
                ]

            # 2. SC amplitudes normalised by window mean (62)
            sc_norm = sc_win / sc_global_mean
            feats.extend(np.mean(sc_norm, axis=0).tolist())
            feats.extend(np.std(sc_norm,  axis=0).tolist())

            # 3. Spike features on variance_turb (5)
            vt      = agg_win[:, 2]
            vt_mean = float(np.mean(vt))
            vt_max  = float(np.max(vt))
            vt_std  = float(np.std(vt))
            feats += [
                vt_max,
                vt_std,
                float(np.sum(vt > vt_mean + 2 * vt_std)),
                float(vt_max / (vt_mean + 1e-6)),
            ]

            # 4. Temporal trend (1)
            half = seq_len // 2
            feats.append(float(np.mean(vt[half:]) - np.mean(vt[:half])))

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

    # ------------------------------------------------------------------ #
    #  SESSION-LEVEL TRAIN/TEST SPLIT                                     #
    # ------------------------------------------------------------------ #
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

    window_feature_names = (
        [f"{f}_{s}" for f in aggregate_features for s in ['mean', 'std', 'max', 'p25', 'p75']]
        + [f"{c}_wmean" for c in valid_sc_cols]
        + [f"{c}_wstd"  for c in valid_sc_cols]
        + ['vt_max', 'vt_std', 'vt_spike_count', 'vt_snr', 'vt_trend']
    )

    return X_train, X_test, y_train, y_test, window_feature_names, label_mapping, groups_train


def train_random_forest(X_train, X_test, y_train, y_test,
                        feature_names, label_mapping, groups_train):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    total = len(X_train) + len(X_test)
    print(f"{'='*80}")
    print(f"TRAIN/TEST SPLIT")
    print(f"{'='*80}")
    print(f"Training set size: {len(X_train)} samples ({len(X_train)/total*100:.1f}%)")
    print(f"Test set size:     {len(X_test)} samples ({len(X_test)/total*100:.1f}%)")

    idx_to_label = {idx: label for label, idx in label_mapping.items()}

    # ------------------------------------------------------------------ #
    #  HYPERPARAMETER SEARCH — GroupKFold prevents session leakage        #
    # ------------------------------------------------------------------ #
    param_dist = {
        'n_estimators':      randint(100, 500),
        'max_depth':         [8, 10, 12, 15, 20, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf':  randint(1, 10),
        'max_features':      ['sqrt', 'log2', 0.3, 0.5],
    }

    cv = GroupKFold(n_splits=5)

    print(f"\n{'='*80}")
    print("HYPERPARAMETER SEARCH (RandomizedSearchCV + GroupKFold)")
    print(f"{'='*80}")
    print(f"  Search iterations : 50")
    print(f"  CV folds          : 5 (session-aware)")
    print(f"  Scoring           : balanced_accuracy")
    print(f"  n_jobs            : -1 (all cores)\n")

    search = RandomizedSearchCV(
        RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
        ),
        param_distributions=param_dist,
        n_iter=50,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    search.fit(X_train, y_train, groups=groups_train)

    print(f"\n  Best CV balanced_accuracy : {search.best_score_:.4f}")
    print(f"  Best parameters:")
    for k, v in sorted(search.best_params_.items()):
        print(f"    {k:22s}: {v}")

    best_model = search.best_estimator_

    # ------------------------------------------------------------------ #
    #  FINAL EVALUATION ON HELD-OUT TEST SESSIONS                        #
    # ------------------------------------------------------------------ #
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

    feature_importances = best_model.feature_importances_
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

    return best_model, label_mapping, scaler


def main():
    print(f"\n{'='*80}")
    print("ESPectre WiFi Sensing - Random Forest Classifier Training")
    print(f"{'='*80}\n")

    X_train, X_test, y_train, y_test, feature_names, label_mapping, groups_train = \
        load_data_from_directories('csv_data')

    
    vt_mean_idx = 15
    print("\nVariance Turb Mean — distribution check:")
    print(f"{'Class':12s}  {'Split':6s}  {'mean':8s}  {'p25':8s}  {'p75':8s}  {'max':8s}")
    for split_name, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
        for idx, label in sorted((v, k) for k, v in label_mapping.items()):
            mask = y == idx
            vals = X[mask, vt_mean_idx]
            print(f"  {label:12s}  {split_name:6s}  "
                f"{np.mean(vals):8.4f}  {np.percentile(vals,25):8.4f}  "
                f"{np.percentile(vals,75):8.4f}  {np.max(vals):8.4f}")
    
    trained_model, label_map, scaler = train_random_forest(
        X_train, X_test, y_train, y_test, feature_names, label_mapping, groups_train)

    
    
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
