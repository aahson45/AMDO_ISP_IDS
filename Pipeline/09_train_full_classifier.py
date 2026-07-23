"""
09_train_full_classifier.py
------------------------------
Phase 2, Step 9: Tier 3 - Full Multi-Class classifier.

Trains BOTH XGBoost and LightGBM (the architecture diagram allows
either) on the six-class taxonomy, using Macro-F1 as the primary
metric - per the thesis's stated evaluation criteria, since Macro-F1
weighs every class equally regardless of size, which matters here:
a model that nails DoS_DDoS (2.2M rows) but completely misses
WEB_ATTACK (20,000 rows) should NOT score well just because the
majority class is easy.

Same train/val/test discipline as 07: val.csv only used for early
stopping, test.csv touched exactly once, at the end.

Input:  splits/train.csv, splits/val.csv, splits/test.csv
Output: two trained models (XGBoost + LightGBM) + a comparison report
        + a machine-readable JSON summary (for downstream slide/table
        generation, so real numbers never need to be hand-copied out
        of a .txt report)
"""

import json
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import os

# ============================================================
# 1. CONFIG
# ============================================================
BASE_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software"
SPLIT_DIR = os.path.join(BASE_DIR, "Data Set", "splits")
TRAIN_CSV = os.path.join(SPLIT_DIR, "train.csv")
VAL_CSV = os.path.join(SPLIT_DIR, "val.csv")
TEST_CSV = os.path.join(SPLIT_DIR, "test.csv")

MODEL_DIR = os.path.join(BASE_DIR, "Data Set", "models")
REPORT_DIR = os.path.join(BASE_DIR, "Data Set", "report")
RESULTS_DIR = os.path.join(BASE_DIR, "Data Set", "processed", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

XGB_MODEL_PATH = os.path.join(MODEL_DIR, "tier3_full_classifier_xgboost.json")
LGB_MODEL_PATH = os.path.join(MODEL_DIR, "tier3_full_classifier_lightgbm.txt")
OUTPUT_REPORT = os.path.join(REPORT_DIR, "09_tier3_full_classifier_report.txt")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "tier3_cicids_classifier_comparison.json")

GOLDEN_12 = [
    "Fwd Header Length", "Bwd Header Length", "Fwd PSH Flags", "ACK Flag Count",
    "Init_Win_bytes_forward", "Flow Duration", "Flow Packets/s", "Flow Bytes/s",
    "Fwd Packet Length Max", "Flow IAT Mean", "Fwd IAT Max", "Bwd IAT Min",
]
LABEL_COL = "unified_label"
RANDOM_SEED = 42
EARLY_STOPPING_ROUNDS = 20


def evaluate(y_true, y_pred, class_names, model_name, report_lines):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    macro_prec = precision_score(y_true, y_pred, average="macro")
    macro_rec = recall_score(y_true, y_pred, average="macro")

    report_lines.append(f"\n=== {model_name} - TEST SET RESULTS ===")
    report_lines.append(f"Accuracy:        {acc:.4f}")
    report_lines.append(f"Macro-F1:        {macro_f1:.4f}  <-- PRIMARY METRIC (equal weight per class)")
    report_lines.append(f"Weighted-F1:     {weighted_f1:.4f}  (for comparison - dominated by large classes)")
    report_lines.append(f"Macro-Precision: {macro_prec:.4f}")
    report_lines.append(f"Macro-Recall:    {macro_rec:.4f}")
    report_lines.append(f"\nPer-class report:\n{classification_report(y_true, y_pred, target_names=class_names, digits=4)}")
    report_lines.append(f"Confusion matrix (rows=true, cols=predicted):\n{class_names}")
    report_lines.append(str(confusion_matrix(y_true, y_pred)))

    per_class_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class_report": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_lines = ["ADMO THESIS - TIER 3 FULL MULTI-CLASS CLASSIFIER REPORT"]

    # --- Load splits ---
    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    val_df = pd.read_csv(VAL_CSV, low_memory=False)
    test_df = pd.read_csv(TEST_CSV, low_memory=False)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df[LABEL_COL])
    y_val = encoder.transform(val_df[LABEL_COL])
    y_test = encoder.transform(test_df[LABEL_COL])
    class_names = list(encoder.classes_)

    X_train, X_val, X_test = train_df[GOLDEN_12], val_df[GOLDEN_12], test_df[GOLDEN_12]

    report_lines.append(f"\nClasses (encoded 0-{len(class_names)-1}): {class_names}")
    report_lines.append(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    results = {}

    # ============================================================
    # XGBoost
    # ============================================================
    report_lines.append("\n\n" + "=" * 70)
    report_lines.append("TRAINING XGBOOST")
    report_lines.append("=" * 70)

    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        eval_metric="mlogloss",
        max_depth=8,
        learning_rate=0.1,
        n_estimators=400,
        random_state=RANDOM_SEED,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    report_lines.append(f"Training stopped at boosting round: {xgb_model.best_iteration}")

    y_pred_xgb = xgb_model.predict(X_test)
    results["XGBoost"] = evaluate(y_test, y_pred_xgb, class_names, "XGBoost", report_lines)

    xgb_model.save_model(XGB_MODEL_PATH)
    report_lines.append(f"\nXGBoost model saved to: {XGB_MODEL_PATH}")

    # ============================================================
    # LightGBM
    # ============================================================
    report_lines.append("\n\n" + "=" * 70)
    report_lines.append("TRAINING LIGHTGBM")
    report_lines.append("=" * 70)

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    lgb_params = {
        "objective": "multiclass",
        "num_class": len(class_names),
        "metric": "multi_logloss",
        "max_depth": 8,
        "learning_rate": 0.1,
        "verbosity": -1,
        "seed": RANDOM_SEED,
    }
    lgb_model = lgb.train(
        lgb_params, lgb_train, num_boost_round=400, valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    report_lines.append(f"Training stopped at boosting round: {lgb_model.best_iteration}")

    y_pred_lgb_proba = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    y_pred_lgb = np.argmax(y_pred_lgb_proba, axis=1)
    results["LightGBM"] = evaluate(y_test, y_pred_lgb, class_names, "LightGBM", report_lines)

    lgb_model.save_model(LGB_MODEL_PATH)
    report_lines.append(f"\nLightGBM model saved to: {LGB_MODEL_PATH}")

    # ============================================================
    # Head-to-head comparison
    # ============================================================
    report_lines.append("\n\n" + "=" * 70)
    report_lines.append("HEAD-TO-HEAD COMPARISON (test set)")
    report_lines.append("=" * 70)
    comparison_df = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk in
                                       ("accuracy", "macro_f1", "weighted_f1")} for k, v in results.items()}).T
    report_lines.append(comparison_df.to_string())

    winner = comparison_df["macro_f1"].idxmax()
    report_lines.append(f"\nHigher Macro-F1: {winner} ({comparison_df.loc[winner, 'macro_f1']:.4f})")
    report_lines.append("(Macro-F1 is the deciding metric per thesis evaluation criteria - "
                          "not accuracy, which would be misleading given class imbalance.)")

    report_text = "\n".join(report_lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)
    print(f"\nReport saved to: {OUTPUT_REPORT}")

    # ============================================================
    # Save machine-readable JSON summary
    # ============================================================
    json_summary = {
        "xgboost": results["XGBoost"],
        "lightgbm": results["LightGBM"],
        "winner_by_macro_f1": winner,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(json_summary, f, indent=4)
    print(f"[+] Saved machine-readable comparison summary to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
