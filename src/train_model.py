import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
import xgboost as xgb
import joblib


def prepare_data(df):
    """
    Prepare the feature dataset for model training:
    drop raw sparse lab columns, drop identifier/non-feature columns,
    split into features (X) and label (y).
    """
    raw_labs = ['Lactate', 'Creatinine', 'WBC', 'BUN',
                'Platelets', 'Bilirubin_total', 'Glucose']

    non_feature_cols = ['Patient_ID', 'SepsisLabel',
                        'Unit1', 'Unit2', 'ICULOS', 'HospAdmTime', 'Hour']

    drop_cols = raw_labs + non_feature_cols

    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['SepsisLabel']

    # Fill remaining NaNs (other raw labs we didn't engineer) with 0
    X = X.fillna(0)

    return X, y


def train_isolation_forest(X_train):
    """
    Train an unsupervised Isolation Forest on the feature set.
    No labels used - it learns what 'normal' looks like and flags deviations.
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=0.0727,  # matches our patient-level sepsis rate from EDA
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Train a supervised XGBoost classifier using SepsisLabel as ground truth.
    Uses scale_pos_weight to handle the severe class imbalance.
    """
    # Calculate class imbalance ratio for weighting
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_xgboost(model, X_test, y_test):
    """
    Evaluate XGBoost using AUC-ROC and precision-recall.
    Accuracy is intentionally NOT used - meaningless with 98% class imbalance.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC: {auc:.4f}")

    y_pred = (y_pred_proba >= 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['No Sepsis', 'Sepsis']))

    return auc


def find_optimal_threshold(model, X_test, y_test, min_recall=0.85):
    """
    Find the threshold that maximizes precision while keeping recall
    above a clinically-acceptable minimum. Prioritizes not missing sepsis
    cases over minimizing false alarms, since missed sepsis is the costlier error.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(
        y_test, y_pred_proba)

    # precision_recall_curve returns one more point than thresholds
    # (it includes the threshold=0 edge case), so trim to match
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # Only consider thresholds where recall meets our clinical minimum
    valid = recalls >= min_recall

    if not valid.any():
        print(
            f"No threshold achieves recall >= {min_recall}. Using default 0.5.")
        return 0.5

    # Among valid thresholds, pick the one with highest precision
    valid_precisions = np.where(valid, precisions, -1)
    best_idx = np.argmax(valid_precisions)
    best_threshold = thresholds[best_idx]

    print(f"Best threshold (recall >= {min_recall}): {best_threshold:.4f}")
    print(
        f"At this threshold - Precision: {precisions[best_idx]:.4f}, Recall: {recalls[best_idx]:.4f}")

    return best_threshold


def evaluate_isolation_forest(model, X_test, y_test):
    """
    Evaluate Isolation Forest by comparing its anomaly flags against
    the true SepsisLabel, even though it never saw labels during training.
    """
    # IsolationForest returns -1 for anomaly, 1 for normal - convert to 0/1
    predictions = model.predict(X_test)
    predictions = np.where(predictions == -1, 1, 0)  # 1 = anomaly/sepsis-like

    # Use decision_function scores for AUC (continuous, not just binary flag)
    # negate so higher = more anomalous
    scores = -model.decision_function(X_test)

    auc = roc_auc_score(y_test, scores)
    print(f"AUC-ROC: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions,
          target_names=['No Sepsis', 'Sepsis']))

    return auc


if __name__ == "__main__":
    # Set the experiment name - MLflow groups runs under experiments
    mlflow.set_experiment("sepsis-detection")

    print("Loading data...")
    df = pd.read_csv('../data/sepsis_features_full.csv')

    print("Preparing features...")
    X, y = prepare_data(df)
    print(f"Feature matrix shape: {X.shape}")

    print("Splitting train/test by patient...")
    patient_sepsis = df.groupby('Patient_ID')['SepsisLabel'].max()

    train_patients, test_patients = train_test_split(
        patient_sepsis.index,
        test_size=0.2,
        random_state=42,
        stratify=patient_sepsis.values
    )

    train_mask = df['Patient_ID'].isin(train_patients)
    test_mask = df['Patient_ID'].isin(test_patients)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ── Isolation Forest Run ──────────────────────────────────────────────────
    print("\n--- Training Isolation Forest ---")
    with mlflow.start_run(run_name="isolation_forest"):
        # Log parameters
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("contamination", 0.0727)
        mlflow.log_param("n_train_patients", len(train_patients))
        mlflow.log_param("n_test_patients", len(test_patients))

        iso_model = train_isolation_forest(X_train)
        iso_auc = evaluate_isolation_forest(iso_model, X_test, y_test)

        # Log metrics
        mlflow.log_metric("auc_roc", iso_auc)

        # Save model artifact to MLflow
        joblib.dump(iso_model, '../models/iso_forest.joblib')
        mlflow.log_artifact('../models/iso_forest.joblib')

        print(f"MLflow run logged: IsolationForest AUC={iso_auc:.4f}")

    # ── XGBoost Run ──────────────────────────────────────────────────────────
    print("\n--- Training XGBoost ---")
    with mlflow.start_run(run_name="xgboost"):
        # Log parameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("n_train_patients", len(train_patients))
        mlflow.log_param("n_test_patients", len(test_patients))
        mlflow.log_param("split_strategy", "patient_level")
        mlflow.log_param("features_removed", "ICULOS,HospAdmTime,Hour")

        xgb_model = train_xgboost(X_train, y_train)
        xgb_auc = evaluate_xgboost(xgb_model, X_test, y_test)

        # Log metrics
        mlflow.log_metric("auc_roc", xgb_auc)

        # Threshold tuning
        best_threshold = find_optimal_threshold(xgb_model, X_test, y_test)
        mlflow.log_metric("optimal_threshold", best_threshold)
        mlflow.log_metric("min_recall_target", 0.85)

        # Log model natively via mlflow.xgboost
        mlflow.xgboost.log_model(xgb_model, "xgb_model")

        # Also save joblib for the API to load
        joblib.dump(xgb_model, '../models/xgb_model.joblib')
        mlflow.log_artifact('../models/xgb_model.joblib')

        print(f"MLflow run logged: XGBoost AUC={xgb_auc:.4f}")

    print("\n--- Summary ---")
    print(f"Isolation Forest AUC: {iso_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    print("\nDone. Run 'mlflow ui' to view results.")
