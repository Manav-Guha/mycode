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

    # Deduplicate targets with identical label vectors — these are
    # artefacts of the label extraction (different pattern titles
    # matching the same set of findings).  Keep the one with the
    # highest confirmed_count.
    seen_vectors: dict[bytes, int] = {}  # hash → index in viable_cols
    dedup_keep: list[int] = []
    dedup_dropped: list[str] = []
    for i, col in enumerate(viable_cols):
        key = Y_viable[:, i].tobytes()
        if key in seen_vectors:
            existing_idx = seen_vectors[key]
            existing_col = viable_cols[existing_idx]
            # Keep the one with higher confirmed_count
            existing_count = target_info.get(existing_col, {}).get("confirmed_count", 0)
            this_count = target_info.get(col, {}).get("confirmed_count", 0)
            if this_count > existing_count:
                # Replace existing with this one
                dedup_dropped.append(existing_col)
                dedup_keep = [j if j != existing_idx else i for j in dedup_keep]
                seen_vectors[key] = i
            else:
                dedup_dropped.append(col)
        else:
            seen_vectors[key] = i
            dedup_keep.append(i)

    if dedup_dropped:
        print(f"  Deduplicated (identical label vectors): {len(dedup_dropped)} targets removed")
        for dc in dedup_dropped:
            print(f"    - {dc}")
        dropped_cols.extend(dedup_dropped)
        viable_cols = [viable_cols[i] for i in dedup_keep]
        Y_viable = Y_viable[:, dedup_keep]
        print(f"  Remaining targets after dedup: {len(viable_cols)}")

    if len(viable_cols) == 0:
        print("ERROR: No viable targets. Aborting.")
        sys.exit(1)

    # Train with per-target class weighting: scale_pos_weight compensates
    # for class imbalance so the model doesn't default to predicting 0.
    print("\nTraining XGBoost multi-label model (class-weighted)...")
    n_samples = X.shape[0]
    estimators = []
    for i in range(Y_viable.shape[1]):
        n_pos = int(Y_viable[:, i].sum())
        n_neg = n_samples - n_pos
        spw = n_neg / max(n_pos, 1)
        est = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        estimators.append(est)

    # MultiOutputClassifier doesn't support per-target params, so we
    # train each estimator individually.
    for i, est in enumerate(estimators):
        est.fit(X, Y_viable[:, i])

    # Wrap into a compatible object for saving
    model = MultiOutputClassifier(XGBClassifier(**XGB_PARAMS))
    model.estimators_ = estimators
    model.classes_ = [np.array([0, 1])] * len(estimators)
    print("  Training complete.")

    # Cross-validated evaluation (also class-weighted)
    print("\nRunning 5-fold cross-validation (class-weighted)...")
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    Y_scores = np.zeros((n_samples, Y_viable.shape[1]), dtype=np.float64)

    for i in range(Y_viable.shape[1]):
        y_col = Y_viable[:, i]
        n_pos = int(y_col.sum())
        n_neg = n_samples - n_pos
        spw = n_neg / max(n_pos, 1)
        for train_idx, test_idx in kf.split(X, y_col):
            est = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
            est.fit(X[train_idx], y_col[train_idx])
            Y_scores[test_idx, i] = est.predict_proba(X[test_idx])[:, 1]

    # Per-target threshold calibration: find the threshold maximising F1
    # on the CV probabilities, rather than assuming 0.5.
    per_target_thresholds = {}
    for i, col in enumerate(viable_cols):
        y_true = Y_viable[:, i]
        y_score = Y_scores[:, i]
        best_f1, best_thresh = 0.0, 0.5
        for thresh in np.arange(0.05, 0.96, 0.01):
            pred = (y_score >= thresh).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            if f > best_f1:
                best_f1, best_thresh = f, float(thresh)
        per_target_thresholds[col] = round(best_thresh, 2)

    # Binary predictions using calibrated thresholds
    Y_pred_cv = np.zeros_like(Y_viable)
    for i, col in enumerate(viable_cols):
        Y_pred_cv[:, i] = (Y_scores[:, i] >= per_target_thresholds[col]).astype(int)

    # Compute per-target AUC, precision, recall, F1
    per_target_auc = {}
    per_target_metrics = {}
    aucs = []
    precisions = []
    recalls = []
    f1s = []
    print("\n  Per-target metrics (calibrated thresholds):")
    print(f"  {'':2s} {'AUC':>5s}  {'Prec':>5s}  {'Rec':>5s}  {'F1':>5s}  {'Thr':>5s}  {'Pos':>5s}  {'Title'}")
    for i, col in enumerate(viable_cols):
        y_true = Y_viable[:, i]
        y_score = Y_scores[:, i]
        y_pred = Y_pred_cv[:, i]
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        title = target_info.get(col, {}).get("title", col)[:40]
        thresh = per_target_thresholds[col]
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
        per_target_metrics[col] = {
            "auc": round(auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "threshold": thresh,
            "n_positive": n_pos,
        }
        marker = "  " if auc >= 0.60 else "!!"
        print(f"  {marker} {auc:.3f}  {prec:.3f}  {rec:.3f}  {f1:.3f}  {thresh:.2f}  {n_pos:5d}  {title}")

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

    # Save metadata (includes per-target calibrated thresholds)
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
        "mean_precision": round(mean_prec, 4),
        "mean_recall": round(mean_rec, 4),
        "mean_f1": round(mean_f1, 4),
        "per_target_auc": per_target_auc,
        "per_target_thresholds": per_target_thresholds,
        "per_target_metrics": per_target_metrics,
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
