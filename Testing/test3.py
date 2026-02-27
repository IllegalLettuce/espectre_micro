import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path


def load_and_prepare_data(baseline_csv, q1_csv, q2_csv):
    """
    Load features from separate CSV files and prepare for training.
    
    Args:
        baseline_csv: Path to baseline CSV file
        q1_csv: Path to Q1 walking CSV file
        q2_csv: Path to Q2 walking CSV file
    
    Returns:
        X: Feature matrix (numpy array)
        y: Labels (numpy array)
        feature_columns: List of feature names
    """
    
    # Load each CSV
    df_baseline = pd.read_csv(baseline_csv)
    df_q1 = pd.read_csv(q1_csv)
    df_q2 = pd.read_csv(q2_csv)
    
    print(f"Loaded {len(df_baseline)} baseline samples from {baseline_csv}")
    print(f"Loaded {len(df_q1)} Q1 samples from {q1_csv}")
    print(f"Loaded {len(df_q2)} Q2 samples from {q2_csv}")
    
    # Verify labels match expected values
    assert df_baseline['label'].iloc[0] == 'baseline', "First CSV should contain 'baseline' labels"
    assert df_q1['label'].iloc[0] in ['q1', 'walking_q1'], "Second CSV should contain 'q1' or 'walking_q1' labels"
    assert df_q2['label'].iloc[0] in ['q2', 'walking_q2'], "Third CSV should contain 'q2' or 'walking_q2' labels"
    
    # Standardize labels
    df_baseline['label'] = 'baseline'
    df_q1['label'] = 'walking_q1'
    df_q2['label'] = 'walking_q2'
    
    # Concatenate all dataframes
    dataframe = pd.concat([df_baseline, df_q1, df_q2], ignore_index=True)
    
    print(f"\nTotal samples: {len(dataframe)}")
    print(f"Columns: {list(dataframe.columns)}")
    
    # Feature columns
    feature_columns = [
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
        'amp_mean_high'
    ]
    
    # Extract features
    X = dataframe[feature_columns].values
    
    # Map labels to integers
    label_mapping = {
        'baseline': 0,
        'walking_q1': 1,
        'walking_q2': 2
    }
    y = dataframe['label'].map(label_mapping).values
    
    # Print summary
    print(f"\n{'='*60}")
    print("Data Summary:")
    print(f"{'='*60}")
    print(f"Total samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"\nClass distribution:")
    
    for label_name, label_value in label_mapping.items():
        count = np.sum(y == label_value)
        percentage = (count / len(y)) * 100
        print(f"  {label_name:12s}: {count:5d} samples ({percentage:.1f}%)")
    
    print(f"{'='*60}")
    
    return X, y, feature_columns


def load_multiple_sessions(session_files):
    """
    Load features from multiple collection sessions (e.g., 5 sessions).
    
    Args:
        session_files: List of tuples [(baseline_csv, q1_csv, q2_csv), ...]
    
    Returns:
        X: Combined feature matrix
        y: Combined labels
        feature_columns: List of feature names
    """
    
    all_dataframes = []
    
    for session_idx, (baseline_csv, q1_csv, q2_csv) in enumerate(session_files):
        print(f"\n--- Loading Session {session_idx + 1} ---")
        
        df_baseline = pd.read_csv(baseline_csv)
        df_q1 = pd.read_csv(q1_csv)
        df_q2 = pd.read_csv(q2_csv)
        
        print(f"  Baseline: {len(df_baseline)} samples")
        print(f"  Q1:       {len(df_q1)} samples")
        print(f"  Q2:       {len(df_q2)} samples")
        
        # Standardize labels
        df_baseline['label'] = 'baseline'
        df_q1['label'] = 'walking_q1'
        df_q2['label'] = 'walking_q2'
        
        # Add session identifier (optional, for analysis)
        df_baseline['session'] = session_idx
        df_q1['session'] = session_idx
        df_q2['session'] = session_idx
        
        # Combine session data
        session_df = pd.concat([df_baseline, df_q1, df_q2], ignore_index=True)
        all_dataframes.append(session_df)
    
    # Concatenate all sessions
    dataframe = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(f"Combined {len(session_files)} sessions")
    print(f"Total samples: {len(dataframe)}")
    print(f"{'='*60}")
    
    # Feature columns
    feature_columns = [
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
        'amp_mean_high'
    ]
    
    # Extract features
    X = dataframe[feature_columns].values
    
    # Map labels
    label_mapping = {
        'baseline': 0,
        'walking_q1': 1,
        'walking_q2': 2
    }
    y = dataframe['label'].map(label_mapping).values
    
    # Print summary
    print(f"\nClass distribution:")
    for label_name, label_value in label_mapping.items():
        count = np.sum(y == label_value)
        percentage = (count / len(y)) * 100
        print(f"  {label_name:12s}: {count:5d} samples ({percentage:.1f}%)")
    
    print(f"{'='*60}")
    
    return X, y, feature_columns


def train_random_forest(X, y, feature_names):
    """Train Random Forest classifier and evaluate performance."""
    
    # Stratified split ensures balanced classes in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train/test distribution
    print(f"\nTraining set class distribution:")
    for label_value in [0, 1, 2]:
        count = np.sum(y_train == label_value)
        print(f"  Class {label_value}: {count} samples")
    
    # Random Forest with good defaults for small datasets
    random_forest_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining Random Forest...")
    random_forest_classifier.fit(X_train, y_train)
    
    # Predictions
    predictions = random_forest_classifier.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"{'='*60}")
    
    # Classification report
    class_names = ['Baseline', 'Walking_Q1', 'Walking_Q2']
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=class_names))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    confusion = confusion_matrix(y_test, predictions)
    print(f"\n{'':15s} Predicted:")
    print(f"{'':15s} {'Baseline':>12s} {'Walking_Q1':>12s} {'Walking_Q2':>12s}")
    print(f"{'Actual:':15s}")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:13s} {confusion[i][0]:12d} {confusion[i][1]:12d} {confusion[i][2]:12d}")
    
    # Feature importances
    feature_importances = random_forest_classifier.feature_importances_
    importance_indices = np.argsort(feature_importances)[::-1]
    
    print(f"\n{'='*60}")
    print("Feature Importances (Ranked):")
    print(f"{'='*60}")
    for i in range(len(feature_names)):
        feature_index = importance_indices[i]
        feature_name = feature_names[feature_index]
        importance = feature_importances[feature_index]
        percentage = importance * 100
        print(f"{i+1:2d}. {feature_name:15s}: {importance:.4f} ({percentage:5.1f}%)")
    
    return random_forest_classifier


def main():
    print("="*60)
    print("Random Forest CSI Classifier - Spatial Localization")
    print("="*60)
    print()

    session_files = [
        ('csi_training_data_baseline.csv', 'csi_training_data_q1.csv', 'csi_training_data_q2.csv')
    ]

    features, labels, feature_names = load_multiple_sessions(session_files)
    trained_model = train_random_forest(features, labels, feature_names)
    
    print("\n" + "="*60)
    print("✓ Random Forest model trained successfully!")
    print("="*60)
    
    # Save model for deployment
    import joblib
    joblib.dump(trained_model, 'rf_spatial_classifier.pkl')
    print("Model saved to 'rf_spatial_classifier.pkl'")
    
    return trained_model


if __name__ == '__main__':
    model = main()
