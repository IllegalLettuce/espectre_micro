import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path


# 802.11n HT20 subcarriers that carry NO information:
# DC subcarrier (index 0), pilot subcarriers, and guard band nulls.
# These are always zero and must be excluded from training.
NULL_SUBCARRIER_INDICES = {0, 1, 2, 3, 4, 5, 27, 28, 29, 30, 31, 32, 33, 59, 60, 61, 62, 63}


def get_sc_feature_columns(dataframe):
    """
    Identify all per-subcarrier amplitude columns (sc_amp_0 ... sc_amp_63)
    present in the dataframe, then exclude known null/pilot/DC subcarriers
    and any column with zero variance across the dataset.

    Returns:
        valid_sc_cols  : list of sc_amp_N column names that carry real signal
        dropped_sc_cols: list of sc_amp_N column names that were dropped
    """
    # All sc_amp_* columns present in the CSV
    all_sc_cols = sorted(
        [c for c in dataframe.columns if c.startswith('sc_amp_')],
        key=lambda c: int(c.split('_')[-1])
    )

    if not all_sc_cols:
        return [], []  # No per-subcarrier features in this CSV

    dropped = []
    valid = []

    for col in all_sc_cols:
        idx = int(col.split('_')[-1])

        # Rule 1: Known 802.11n null / DC / guard-band indices
        if idx in NULL_SUBCARRIER_INDICES:
            dropped.append((col, 'known null subcarrier (802.11n HT20)'))
            continue

        # Rule 2: Zero variance — constant column carries no information
        if dataframe[col].std() == 0:
            dropped.append((col, f'zero variance (all values = {dataframe[col].iloc[0]:.1f})'))
            continue

        valid.append(col)

    return valid, dropped


def load_data_from_directories(base_dir='csv_data'):
    """
    Automatically scan directories and load all CSV files.

    Directory structure expected:
        csv_data/
            baseline/      *.csv
            movement_q1/   *.csv
            movement_q2/   *.csv

    Returns:
        X              : Feature matrix (numpy array)
        y              : Labels (numpy array)
        feature_columns: List of feature names used
        label_mapping  : Dict mapping label name → integer
    """

    base_path = Path(base_dir)

    directory_mappings = {
        'baseline':     'Baseline',
        'movement_q1':  'Quadrant_1',
        'movement_q2':  'Quadrant_2',
    }

    all_dataframes = []

    print(f"{'='*80}")
    print(f"Loading data from: {base_path.absolute()}")
    print(f"{'='*80}\n")

    for dir_name, label_name in directory_mappings.items():
        dir_path = base_path / dir_name

        if not dir_path.exists():
            print(f"⚠️  Warning: Directory '{dir_path}' not found, skipping...")
            continue

        csv_files = list(dir_path.glob('*.csv'))

        if not csv_files:
            print(f"⚠️  Warning: No CSV files found in '{dir_path}', skipping...")
            continue

        print(f"📁 {dir_name}/ → Label: '{label_name}'")
        print(f"   Found {len(csv_files)} CSV file(s)")

        session_dfs = []
        total_samples = 0

        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                df['label'] = label_name
                df['source_file'] = csv_file.name
                session_dfs.append(df)
                total_samples += len(df)
                print(f"      ✓ {csv_file.name}: {len(df)} samples")
            except Exception as e:
                print(f"      ✗ Error loading {csv_file.name}: {e}")

        if session_dfs:
            combined_df = pd.concat(session_dfs, ignore_index=True)
            all_dataframes.append(combined_df)
            print(f"   Total: {total_samples} samples for '{label_name}'\n")

    if not all_dataframes:
        raise ValueError("No data loaded! Check your directory structure and CSV files.")

    dataframe = pd.concat(all_dataframes, ignore_index=True)

    print(f"{'='*80}")
    print(f"COMBINED DATASET")
    print(f"{'='*80}")
    print(f"Total samples loaded: {len(dataframe)}")

    print(f"\nClass distribution:")
    label_counts = dataframe['label'].value_counts()
    for lname, count in label_counts.items():
        percentage = (count / len(dataframe)) * 100
        print(f"  {lname:15s}: {count:5d} samples ({percentage:5.1f}%)")

    # ------------------------------------------------------------------ #
    #  AGGREGATE FEATURES  (always present)                               #
    # ------------------------------------------------------------------ #
    aggregate_features = [
        'entropy_turb',
        'iqr_turb',
        'variance_turb',
        'skewness',
        'kurtosis',
        'amp_mean',
        'amp_range',
        'amp_std',
        'amp_mean_low',
        'amp_mean_mid',
        'amp_mean_high',
    ]

    missing = [f for f in aggregate_features if f not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing aggregate features in CSV files: {missing}")

    # ------------------------------------------------------------------ #
    #  PER-SUBCARRIER FEATURES  (filter out null / zero-variance ones)    #
    # ------------------------------------------------------------------ #
    valid_sc_cols, dropped_sc_cols = get_sc_feature_columns(dataframe)

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

    # ------------------------------------------------------------------ #
    #  COMBINED FEATURE SET                                               #
    # ------------------------------------------------------------------ #
    feature_columns = aggregate_features + valid_sc_cols

    print(f"\n{'='*80}")
    print("FEATURE SET SUMMARY")
    print(f"{'='*80}")
    print(f"  Aggregate features  : {len(aggregate_features)}")
    print(f"  Subcarrier features : {len(valid_sc_cols)}")
    print(f"  Total features      : {len(feature_columns)}")

    X = dataframe[feature_columns].values

    unique_labels = sorted(dataframe['label'].unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    print(f"\nLabel encoding:")
    for lname, lidx in label_mapping.items():
        print(f"  {lname:15s} → {lidx}")

    y = dataframe['label'].map(label_mapping).values

    print(f"{'='*80}\n")

    return X, y, feature_columns, label_mapping


def train_random_forest(X, y, feature_names, label_mapping):
    """Train Random Forest classifier and evaluate performance."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"{'='*80}")
    print(f"TRAIN/TEST SPLIT")
    print(f"{'='*80}")
    print(f"Training set size: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set size:     {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    idx_to_label = {idx: label for label, idx in label_mapping.items()}

    print(f"\nTraining set class distribution:")
    for label_idx in sorted(label_mapping.values()):
        count = np.sum(y_train == label_idx)
        print(f"  {idx_to_label[label_idx]:15s}: {count:4d} samples")

    print(f"\nTest set class distribution:")
    for label_idx in sorted(label_mapping.values()):
        count = np.sum(y_test == label_idx)
        print(f"  {idx_to_label[label_idx]:15s}: {count:4d} samples")

    random_forest_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )

#                      Predicted:
# Actual:                    Baseline     Quadrant_1     Quadrant_2
#   Baseline                      502              4              5
#   Quadrant_1                      5            225             31
#   Quadrant_2                      7             21            228
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

    # Feature importances
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

    return random_forest_classifier, label_mapping


def main():
    print(f"\n{'='*80}")
    print("ESPectre WiFi Sensing - Random Forest Classifier Training")
    print(f"{'='*80}\n")

    features, labels, feature_names, label_mapping = load_data_from_directories('csv_data')

    trained_model, label_map = train_random_forest(features, labels, feature_names, label_mapping)

    import joblib

    model_filename = 'rf_spatial_classifier.pkl'
    joblib.dump(trained_model, model_filename)

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
