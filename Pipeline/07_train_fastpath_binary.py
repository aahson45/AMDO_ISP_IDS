"""
07_train_fastpath_binary.py
------------------------------
Phase 2, Step 7: Tier 1 - "Lightning Binary" fast-path classifier.

Target definition (see note in chat - flagged as an assumption to
confirm with supervisor): DoS_DDoS vs EVERYTHING ELSE (not just vs
BENIGN), since a real fast-path filter must triage any incoming
traffic, not just distinguish DDoS from benign specifically.

Trains a binary XGBoost model:
  - train.csv used to fit the model
  - val.csv used as an eval_set for early stopping (stops training
    once validation performance stops improving - prevents overfitting)
  - test.csv used ONLY at the end, for final, honest evaluation

Also reports single-row inference latency as an early sanity check
against the architecture's <0.5ms Tier 1 target - full, rigorous
Type A latency benchmarking (with MP1-MP4 markers) is 10_latency_bench.py's
job; this is just a quick proxy so we're not surprised later.

Input:  splits/train.csv, splits/val.csv, splits/test.csv
Output: trained model file + evaluation report
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import time
import os
import json

# ============================================================
# 1. CONFIG
# ============================================================
SPLIT_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\splits"
TRAIN_CSV = os.path.join(SPLIT_DIR, "train.csv")
VAL_CSV = os.path.join(SPLIT_DIR, "val.csv")
TEST_CSV = os.path.join(SPLIT_DIR, "test.csv")

MODEL_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\models"
REPORT_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\report"

MODEL_PATH = os.path.join(MODEL_DIR, "tier1_fastpath_binary.json")
OUTPUT_REPORT = os.path.join(REPORT_DIR, "07_tier1_fastpath_report.txt")

GOLDEN_12 = [
    "Fwd Header Length", "Bwd Header Length", "Fwd PSH Flags", "ACK Flag Count",
    "Init_Win_bytes_forward", "Flow Duration", "Flow Packets/s", "Flow Bytes/s",
    "Fwd Packet Length Max", "Flow IAT Mean", "Fwd IAT Max", "Bwd IAT Min",
]
LABEL_COL = "unified_label"
POSITIVE_CLASS = "DoS_DDoS"  # this is the "1" class; everything else becomes "0"

RANDOM_SEED = 42

# XGBoost hyperparameters - reasonable, commonly-used defaults for a
# first pass. Not yet tuned - hyperparameter tuning is a separate,
# later exercise once the full pipeline is proven to work end-to-end.
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",  # area under precision-recall curve - better than accuracy for imbalanced binary problems
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_SEED,
    "n_estimators": 300,
}
EARLY_STOPPING_ROUNDS = 20


def make_binary_target(df):
    return (df[LABEL_COL] == POSITIVE_CLASS).astype(int)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_lines = ["ADMO THESIS - TIER 1 FAST-PATH BINARY CLASSIFIER REPORT"]
    report_lines.append(f"Positive class definition: '{POSITIVE_CLASS}' vs everything else (assumption - "
                          f"flag with supervisor if 'vs BENIGN only' was intended instead)")

    # --- Load splits ---
    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    val_df = pd.read_csv(VAL_CSV, low_memory=False)
    test_df = pd.read_csv(TEST_CSV, low_memory=False)

    X_train, y_train = train_df[GOLDEN_12], make_binary_target(train_df)
    X_val, y_val = val_df[GOLDEN_12], make_binary_target(val_df)
    X_test, y_test = test_df[GOLDEN_12], make_binary_target(test_df)

    report_lines.append(f"\nTrain: {len(X_train):,} rows ({y_train.sum():,} positive / {len(y_train)-y_train.sum():,} negative)")
    report_lines.append(f"Val:   {len(X_val):,} rows ({y_val.sum():,} positive / {len(y_val)-y_val.sum():,} negative)")
    report_lines.append(f"Test:  {len(X_test):,} rows ({y_test.sum():,} positive / {len(y_test)-y_test.sum():,} negative)")

    # --- Train, with early stopping on the validation set ---
    model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    report_lines.append(f"\nTraining stopped at boosting round: {model.best_iteration} "
                          f"(early stopping patience: {EARLY_STOPPING_ROUNDS} rounds)")

    # --- Evaluate on the held-out test set (never seen during training/tuning) ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    report_lines.append("\n--- TEST SET RESULTS (held out, never used in training or tuning) ---")
    report_lines.append(f"Accuracy:  {acc:.4f}")
    report_lines.append(f"Precision: {prec:.4f}")
    report_lines.append(f"Recall:    {rec:.4f}")
    report_lines.append(f"F1-score:  {f1:.4f}")
    report_lines.append(f"\nConfusion matrix:\n{cm}")
    report_lines.append(f"\n{classification_report(y_test, y_pred, target_names=['NOT_DDoS', 'DoS_DDoS'])}")

    # --- Quick single-row inference latency proxy (full benchmark is 10_latency_bench.py) ---
    sample = X_test.iloc[[0]]
    n_warmup = 20
    n_timed = 200
    for _ in range(n_warmup):
        model.predict(sample)

    timings = []
    for _ in range(n_timed):
        start = time.perf_counter()
        model.predict(sample)
        timings.append((time.perf_counter() - start) * 1000)  # convert to ms

    mean_latency_ms = float(np.mean(timings))
    p95_latency_ms = float(np.percentile(timings, 95))

    report_lines.append(f"\n--- Single-row inference latency (proxy check, {n_timed} calls after {n_warmup} warmup) ---")
    report_lines.append(f"Mean latency: {mean_latency_ms:.4f} ms")
    report_lines.append(f"P95 latency:  {p95_latency_ms:.4f} ms")
    report_lines.append(f"Architecture target: < 0.5 ms")
    report_lines.append("NOTE: this is a rough proxy on a plain (non-INT8-quantized) model, run from Python "
                          "(not a compiled fast-path). Full Type A benchmarking with MP1-MP4 markers happens "
                          "in 10_latency_bench.py; INT8 quantization for deployment is a separate step for Phase 5.")

    # --- Save model ---
    model.save_model(MODEL_PATH)
    report_lines.append(f"\nModel saved to: {MODEL_PATH}")

    report_text = "\n".join(report_lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)
    print(f"\nReport saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
