import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def extract_features(csi_data):
    """
    Extract statistical features from CSI data packet-by-packet.
    Each packet has shape (128,) representing I/Q pairs for 64 subcarriers.
    """
    features = []
    for packet in csi_data:
        packet_features = []

        # Basic statistics
        packet_features.append(np.mean(packet))
        packet_features.append(np.std(packet))
        packet_features.append(np.var(packet))
        packet_features.append(np.max(packet))
        packet_features.append(np.min(packet))
        packet_features.append(np.median(packet))

        # Range and percentiles
        packet_features.append(np.ptp(packet))  # peak-to-peak (range)
        packet_features.append(np.percentile(packet, 25))
        packet_features.append(np.percentile(packet, 75))

        # Amplitude features (convert I/Q pairs to magnitude)
        i_vals = packet[::2].astype(np.float64)  # In-phase (even indices)
        q_vals = packet[1::2].astype(np.float64)  # Quadrature (odd indices)
        amplitudes = np.sqrt(i_vals**2 + q_vals**2)

        packet_features.append(np.mean(amplitudes))
        packet_features.append(np.std(amplitudes))
        packet_features.append(np.max(amplitudes))
        packet_features.append(np.min(amplitudes))

        features.append(packet_features)

    return np.array(features)

def load_and_prepare_data():
    """Load CSI data from npz files and prepare features and labels."""

    # Define file paths for all datasets
    baseline_files = [
        'dining_baseline/dining_baseline_s3_64sc_20260130_080521.npz',
        # 'dining_baseline/dining_baseline_s3_64sc_20260130_080751.npz',
        # 'dining_baseline/dining_baseline_s3_64sc_20260130_080941.npz',
        # 'dining_baseline/dining_baseline_s3_64sc_20260130_081219.npz',
        # 'dining_baseline/dining_baseline_s3_64sc_20260130_081350.npz',
    ]

    walking_q1_files = [
        'dining_walking_q1/dining_walking_q1_s3_64sc_20260130_081503.npz'
        # 'dining_walking_q1/dining_walking_q1_s3_64sc_20260130_081609.npz',
        # 'dining_walking_q1/dining_walking_q1_s3_64sc_20260130_081709.npz',
        # 'dining_walking_q1/dining_walking_q1_s3_64sc_20260130_081828.npz',
        # 'dining_walking_q1/dining_walking_q1_s3_64sc_20260130_081932.npz',
    ]

    walking_q2_files = [
        # 'dining_walking_q2/dining_walking_q2_s3_64sc_20260130_082043.npz',
        # 'dining_walking_q2/dining_walking_q2_s3_64sc_20260130_082505.npz',
        # 'dining_walking_q2/dining_walking_q2_s3_64sc_20260130_082620.npz',
        # 'dining_walking_q2/dining_walking_q2_s3_64sc_20260130_082804.npz',
        'dining_walking_q2/dining_walking_q2_s3_64sc_20260130_082915.npz',
    ]

    # Label mapping: 0 = baseline, 1 = walking_q1, 2 = walking_q2
    label_baseline = 0
    label_walking_q1 = 1
    label_walking_q2 = 2

    all_features = []
    all_labels = []

    # Load and process baseline files
    print("Loading baseline files...")
    for i, filepath in enumerate(baseline_files):
        npz = np.load(filepath)
        csi_data = npz['csi_data']
        print(f"  [{i+1}/{len(baseline_files)}] {filepath.split('/')[-1]}: {csi_data.shape[0]} packets")

        features = extract_features(csi_data)
        labels = np.full(len(features), label_baseline)

        all_features.append(features)
        all_labels.append(labels)

    baseline_total = sum([len(f) for f in all_features])

    # Load and process walking Q1 files
    print("\nLoading walking Q1 files...")
    for i, filepath in enumerate(walking_q1_files):
        npz = np.load(filepath)
        csi_data = npz['csi_data']
        print(f"  [{i+1}/{len(walking_q1_files)}] {filepath.split('/')[-1]}: {csi_data.shape[0]} packets")

        features = extract_features(csi_data)
        labels = np.full(len(features), label_walking_q1)

        all_features.append(features)
        all_labels.append(labels)

    q1_total = sum([len(f) for f in all_features]) - baseline_total

    # Load and process walking Q2 files
    print("\nLoading walking Q2 files...")
    for i, filepath in enumerate(walking_q2_files):
        npz = np.load(filepath)
        csi_data = npz['csi_data']
        print(f"  [{i+1}/{len(walking_q2_files)}] {filepath.split('/')[-1]}: {csi_data.shape[0]} packets")

        features = extract_features(csi_data)
        labels = np.full(len(features), label_walking_q2)

        all_features.append(features)
        all_labels.append(labels)

    q2_total = sum([len(f) for f in all_features]) - baseline_total - q1_total

    # Combine all data
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    print(f"\n{'='*60}")
    print("Data Summary:")
    print(f"{'='*60}")
    print(f"Total samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"\nClass distribution:")
    print(f"  Baseline:    {baseline_total:5d} packets ({baseline_total/len(X)*100:.1f}%)")
    print(f"  Walking Q1:  {q1_total:5d} packets ({q1_total/len(X)*100:.1f}%)")
    print(f"  Walking Q2:  {q2_total:5d} packets ({q2_total/len(X)*100:.1f}%)")
    print(f"{'='*60}")

    return X, y

def train_random_forest(X, y):
    """Train a Random Forest classifier."""

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Create and train Random Forest
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    rf_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_classifier.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Determine which classes are actually present in the data
    unique_labels = np.unique(y)
    class_names_map = {0: 'Baseline', 1: 'Walking_Q1', 2: 'Walking_Q2'}
    class_names = [class_names_map[label] for label in sorted(unique_labels)]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=sorted(unique_labels), target_names=class_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=sorted(unique_labels))
    print(f"\nLabels: {class_names}")
    print(f"{'':15s} Predicted:")
    print(f"{'':15s} {'Baseline':>10s} {'Walking_Q1':>10s} {'Walking_Q2':>10s}")
    print(f"{'Actual:':15s}")
    print(f"{'  Baseline':15s} {cm[0][0]:10d} {cm[0][1]:10d} {cm[0][2]:10d}")
    print(f"{'  Walking_Q1':15s} {cm[1][0]:10d} {cm[1][1]:10d} {cm[1][2]:10d}")
    print(f"{'  Walking_Q2':15s} {cm[2][0]:10d} {cm[2][1]:10d} {cm[2][2]:10d}")

    # Feature importance
    feature_names = ['mean', 'std', 'var', 'max', 'min', 'median', 'range',
                     'p25', 'p75', 'amp_mean', 'amp_std', 'amp_max', 'amp_min']
    importances = rf_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 10 Feature Importances:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:12s}: {importances[idx]:.4f}")

    return rf_classifier

def main():
    print("="*60)
    print("Random Forest CSI Classifier - Dining Room Dataset")
    print("="*60)
    print()

    # Load and prepare data
    X, y = load_and_prepare_data()

    # Train Random Forest classifier
    model = train_random_forest(X, y)

    print("\n" + "="*60)
    print("✓ Random Forest model trained successfully!")
    print("="*60)

    return model

if __name__ == '__main__':
    model = main()
