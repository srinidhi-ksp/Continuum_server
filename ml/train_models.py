"""
ml/train_models.py

Trains:
  1. RandomForestClassifier  -> predicts load_level (Low/Medium/High)
  2. IsolationForest         -> flags unusual behavioral weeks

Run:
    python ml/train_models.py

Reads:
    ml/data/caregiver_dataset.csv
Writes:
    ml/models/caregiver_load_model.joblib
    ml/models/caregiver_anomaly_model.joblib
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "caregiver_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "tasks_per_day",
    "missed_tasks",
    "night_interruptions",
    "avg_sleep_hours",
    "late_logs",
    "task_share",
    "notifications_received",
    "scenario_attempts",
]
TARGET = "load_level"


def train_random_forest(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n=== Random Forest — Load Level Classifier ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred, average='macro'):.3f}")
    print(f"F1 score : {f1_score(y_test, y_pred, average='macro'):.3f}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    labels = sorted(y.unique())
    print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels))
    print("\nFull report:\n", classification_report(y_test, y_pred))

    print("\nFeature importances (what the model actually relies on):")
    importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(importances.round(3))

    joblib.dump(clf, MODELS_DIR / "caregiver_load_model.joblib")
    print(f"\nSaved -> {MODELS_DIR / 'caregiver_load_model.joblib'}")
    return clf


def train_isolation_forest(df: pd.DataFrame):
    X = df[FEATURES]

    # contamination = expected proportion of "unusual" weeks in the data.
    # 0.08 is a reasonable starting assumption for a prototype; tune later
    # once you see how many real weeks actually look anomalous.
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=42,
    )
    iso.fit(X)

    preds = iso.predict(X)          # -1 = anomaly, 1 = normal
    n_anomalies = int((preds == -1).sum())

    print("\n=== Isolation Forest — Anomaly Detector ===")
    print(f"Flagged {n_anomalies} / {len(df)} rows as anomalous "
          f"({n_anomalies/len(df):.1%})")

    joblib.dump(iso, MODELS_DIR / "caregiver_anomaly_model.joblib")
    print(f"Saved -> {MODELS_DIR / 'caregiver_anomaly_model.joblib'}")
    return iso


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found — run `python ml/generate_dataset.py` first."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    train_random_forest(df)
    train_isolation_forest(df)

    print("\nDone. Both model files are in ml/models/ — ml_router.py loads them from there.")
