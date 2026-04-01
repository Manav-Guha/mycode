#!/usr/bin/env python3
"""Train XGBoost prediction model from corpus training data.

Reads training_data.csv, trains a multi-label XGBoost classifier,
evaluates with cross-validation, saves model + metadata.

Outputs:
  - src/mycode/data/prediction_model.joblib
  - src/mycode/data/model_metadata.json
"""

import datetime
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).parent.parent / "src" / "mycode" / "data"
TRAINING_CSV = DATA_DIR / "training_data.csv"
TARGET_COLUMNS_FILE = DATA_DIR / "target_columns.json"
MODEL_OUTPUT = DATA_DIR / "prediction_model.joblib"
METADATA_OUTPUT = DATA_DIR / "model_metadata.json"

# Minimum positive samples for a target to be included in training.
# Targets below this are dropped (undefined AUC, poor generalization).
MIN_POSITIVE_SAMPLES = 5

# XGBoost hyperparameters — conservative for 4K-20K samples.
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
}


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    if not TRAINING_CSV.exists():
        print(f"ERROR: {TRAINING_CSV} not found. Run build_training_data.py first.")
        sys.exit(1)

    print("Loading training data...")
    df = pd.read_csv(TRAINING_CSV)
    print(f"  {len(df)} samples, {len(df.columns)} columns")

    # Load target column info
    with open(TARGET_COLUMNS_FILE) as f:
        target_meta = json.load(f)
    target_columns = target_meta["target_columns"]
    target_info = target_meta["target_info"]

    # Split features and targets
    feature_columns = [c for c in df.columns if not c.startswith("target_")]
    # Only keep targets that exist in the CSV
    target_columns = [c for c in target_columns if c in df.columns]

    X = df[feature_columns].values.astype(np.float32)
    Y = df[target_columns].values.astype(np.int32)

    print(f"  Features: {X.shape[1]}")
    print(f"  Targets: {Y.shape[1]}")

    # Filter out targets with too few positives
    viable_mask = Y.sum(axis=0) >= MIN_POSITIVE_SAMPLES
    viable_cols = [c for c, v in zip(target_columns, viable_mask) if v]
    dropped_cols = [c for c, v in zip(target_columns, viable_mask) if not v]
    Y_viable = Y[:, viable_mask]

    print(f"  Viable targets (>={MIN_POSITIVE_SAMPLES} positives): {len(viable_cols)}")
    if dropped_cols:
        print(f"  Dropped targets (<{MIN_POSITIVE_SAMPLES} positives): {len(dropped_cols)}")

    if len(viable_cols) == 0:
        print("ERROR: No viable targets. Aborting.")
        sys.exit(1)

    # Train model
    print("\nTraining XGBoost multi-label model...")
    base = XGBClassifier(**XGB_PARAMS)
    model = MultiOutputClassifier(base)
    model.fit(X, Y_viable)
    print("  Training complete.")

    # Cross-validated evaluation
    print("\nRunning 5-fold cross-validation...")
    # predict_proba returns list of (n_samples, 2) arrays
    Y_proba_cv = cross_val_predict(
        MultiOutputClassifier(XGBClassifier(**XGB_PARAMS)),
        X, Y_viable, cv=5, method="predict_proba",
    )

    # Extract probability of positive class from each target's (n, 2) array
    if isinstance(Y_proba_cv, list):
        # MultiOutputClassifier returns list of arrays
        Y_scores = np.column_stack([p[:, 1] for p in Y_proba_cv])
    else:
        Y_scores = Y_proba_cv

    # Binary predictions at 0.5 threshold for precision/recall/F1
    Y_pred_cv = (Y_scores >= 0.5).astype(int)

    # Compute per-target AUC, precision, recall, F1
    per_target_auc = {}
    aucs = []
    precisions = []
    recalls = []
    f1s = []
    print("\n  Per-target metrics:")
    print(f"  {'':2s} {'AUC':>5s}  {'Prec':>5s}  {'Rec':>5s}  {'F1':>5s}  {'Pos':>5s}  {'Title'}")
    for i, col in enumerate(viable_cols):
        y_true = Y_viable[:, i]
        y_score = Y_scores[:, i]
        y_pred = Y_pred_cv[:, i]
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        title = target_info.get(col, {}).get("title", col)[:45]
        if n_pos < 2 or n_neg < 2:
            print(f"    {col}: SKIP (n_pos={n_pos})")
            continue
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            print(f"    {col}: SKIP (AUC undefined)")
            continue
        prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        aucs.append(auc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        per_target_auc[col] = round(auc, 4)
        marker = "  " if auc >= 0.60 else "!!"
        print(f"  {marker} {auc:.3f}  {prec:.3f}  {rec:.3f}  {f1:.3f}  {n_pos:5d}  {title}")

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    mean_prec = float(np.mean(precisions)) if precisions else 0.0
    mean_rec = float(np.mean(recalls)) if recalls else 0.0
    mean_f1 = float(np.mean(f1s)) if f1s else 0.0
    print(f"\n  Mean AUC: {mean_auc:.3f}  Prec: {mean_prec:.3f}  Rec: {mean_rec:.3f}  F1: {mean_f1:.3f}  (across {len(aucs)} targets)")

    if mean_auc < 0.50:
        print("WARNING: Mean AUC below random. Model may not be useful.")

    # Save model
    print(f"\nSaving model to {MODEL_OUTPUT}...")
    joblib.dump(model, MODEL_OUTPUT)
    model_size_mb = MODEL_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"  Model size: {model_size_mb:.1f} MB")

    # Save metadata
    metadata = {
        "feature_columns": feature_columns,
        "target_columns": viable_cols,
        "dropped_targets": dropped_cols,
        "target_info": {
            col: target_info[col]
            for col in viable_cols
            if col in target_info
        },
        "training_samples": len(df),
        "mean_auc": round(mean_auc, 4),
        "per_target_auc": per_target_auc,
        "xgb_params": XGB_PARAMS,
        "sklearn_version": __import__("sklearn").__version__,
        "xgboost_version": __import__("xgboost").__version__,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(METADATA_OUTPUT, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {METADATA_OUTPUT}")

    print("\nDone.")


if __name__ == "__main__":
    main()
