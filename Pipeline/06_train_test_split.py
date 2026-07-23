"""
06_train_test_split.py
------------------------
Phase 2, Step 6: Stratified train/validation/test split.

Splits balanced_dataset.csv (from 05_ctgan_smoteenn_augment.py) into
three files:
  - train.csv (80%) - used to actually fit the models in 07-09
  - val.csv   (10%) - used for early stopping / hyperparameter checks
                       during training (XGBoost/LightGBM both support
                       an eval_set for this)
  - test.csv  (10%) - held out completely until final evaluation;
                       never touched during training or tuning

The split is STRATIFIED by unified_label - each class keeps the same
proportion in all three files as it has in the full dataset, so no
partition is accidentally starved of a rare class by chance.

A fixed random seed (42, same as 05) makes this split exactly
reproducible - re-running this script produces byte-identical splits.

Input:  balanced_dataset.csv
Output: train.csv, val.csv, test.csv + a JSON manifest documenting
        exact counts, ratios, and the seed used (for the thesis
        methodology chapter / reproducibility appendix).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

# ============================================================
# 1. CONFIG
# ============================================================
INPUT_CSV = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\balanced_dataset.csv"

SPLIT_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\splits"
REPORT_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\report"

TRAIN_CSV = os.path.join(SPLIT_DIR, "train.csv")
VAL_CSV = os.path.join(SPLIT_DIR, "val.csv")
TEST_CSV = os.path.join(SPLIT_DIR, "test.csv")
MANIFEST_JSON = os.path.join(REPORT_DIR, "06_split_manifest.json")
OUTPUT_REPORT = os.path.join(REPORT_DIR, "06_train_test_split_report.txt")

RANDOM_SEED = 42  # same seed as 05 - consistency across the pipeline

# Ratios must sum to 1.0
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

LABEL_COL = "unified_label"


def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    report_lines = ["ADMO THESIS - TRAIN/VAL/TEST SPLIT REPORT"]

    # --- Load the full balanced dataset (small enough now - no chunking needed) ---
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    report_lines.append(f"\nTotal rows loaded: {len(df):,}")
    report_lines.append(f"Random seed: {RANDOM_SEED}")
    report_lines.append(f"Split ratios - train: {TRAIN_RATIO}, val: {VAL_RATIO}, test: {TEST_RATIO}")

    # --- Step 1: split off the training set first ---
    # (temp holds what will become val + test combined)
    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        stratify=df[LABEL_COL],
        random_state=RANDOM_SEED,
    )

    # --- Step 2: split temp into val and test, in the correct proportion ---
    # temp is (VAL_RATIO + TEST_RATIO) of the whole dataset; we need val's
    # share OF THAT REMAINDER, not of the original total.
    val_share_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_share_of_temp,
        stratify=temp_df[LABEL_COL],
        random_state=RANDOM_SEED,
    )

    # --- Save the three files ---
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    report_lines.append(f"\ntrain.csv: {len(train_df):,} rows -> {TRAIN_CSV}")
    report_lines.append(f"val.csv:   {len(val_df):,} rows -> {VAL_CSV}")
    report_lines.append(f"test.csv:  {len(test_df):,} rows -> {TEST_CSV}")

    # --- Verify stratification actually held - compare class proportions ---
    report_lines.append("\n--- Stratification check: class % in each split (should closely match) ---")

    full_props = (df[LABEL_COL].value_counts(normalize=True) * 100).round(3)
    train_props = (train_df[LABEL_COL].value_counts(normalize=True) * 100).round(3)
    val_props = (val_df[LABEL_COL].value_counts(normalize=True) * 100).round(3)
    test_props = (test_df[LABEL_COL].value_counts(normalize=True) * 100).round(3)

    comparison = pd.DataFrame({
        "Full_%": full_props,
        "Train_%": train_props,
        "Val_%": val_props,
        "Test_%": test_props,
    }).sort_values("Full_%", ascending=False)

    report_lines.append(comparison.to_string())

    max_deviation = (comparison[["Train_%", "Val_%", "Test_%"]].sub(comparison["Full_%"], axis=0)).abs().max().max()
    report_lines.append(f"\nMax deviation from full-dataset class proportion across all splits/classes: {max_deviation:.3f} percentage points")
    if max_deviation < 0.1:
        report_lines.append("Stratification confirmed: class balance is essentially identical across all three splits.")
    else:
        report_lines.append("NOTE: some deviation present - review the table above.")

    # --- Save manifest for reproducibility documentation ---
    manifest = {
        "input_file": INPUT_CSV,
        "random_seed": RANDOM_SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "row_counts": {
            "total": len(df),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "class_distribution_full_pct": full_props.to_dict(),
        "output_files": {"train": TRAIN_CSV, "val": VAL_CSV, "test": TEST_CSV},
    }
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    report_text = "\n".join(report_lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nManifest saved to: {MANIFEST_JSON}")
    print(f"Report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
