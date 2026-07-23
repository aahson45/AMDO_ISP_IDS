"""
05_ctgan_smoteenn_augment.py
------------------------------
Phase 2, Step 5: Class balancing.

Three stages:
  1. Downsample BENIGN (the overwhelming majority class) via chunked
     random sampling - never loads all 15M+ BENIGN rows into memory.
  2. CTGAN synthesis for classes below CTGAN_THRESHOLD rows - with
     current data, this is ONLY WEB_ATTACK (3,108 real rows). CTGAN
     trains a small generative model on the REAL rows of that class
     and produces synthetic rows that follow the same statistical
     patterns, bringing the class up to CTGAN_TARGET_PER_CLASS.
  3. SMOTEENN cleanup - applied with an explicit sampling_strategy
     that only touches classes still below the target, so we don't
     waste hours running nearest-neighbor search across untouched
     majority classes.

Input:  golden12_features.csv (from 03_feature_selection.py)
Output: balanced_dataset.csv + a report on what was added/removed
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# 1. CONFIG - the knobs that control runtime and behaviour
# ============================================================
INPUT_CSV = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\golden12_features.csv"

REPORT_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\report"
OUTPUT_CSV = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\balanced_dataset.csv"
OUTPUT_REPORT = os.path.join(REPORT_DIR, "05_augmentation_report.txt")

CHUNK_SIZE = 200_000
RANDOM_SEED = 42  # fixed seed - reproducibility matters for a thesis

GOLDEN_12 = [
    "Fwd Header Length", "Bwd Header Length", "Fwd PSH Flags", "ACK Flag Count",
    "Init_Win_bytes_forward", "Flow Duration", "Flow Packets/s", "Flow Bytes/s",
    "Fwd Packet Length Max", "Flow IAT Mean", "Fwd IAT Max", "Bwd IAT Min",
]

# --- Stage 1: BENIGN downsampling ---
BENIGN_DOWNSAMPLE_TARGET = 300_000
# Why 300,000: comparable order of magnitude to the mid-size real classes
# (BRUTE_FORCE ~395K, INFILTRATION ~319K), rather than letting BENIGN
# dominate 80%+ of every training batch as it does at 15.6M rows.

# --- Stage 2: CTGAN synthesis ---
CTGAN_THRESHOLD = 50_000       # classes below this row count get CTGAN
CTGAN_TARGET_PER_CLASS = 20_000  # bring qualifying classes up to this many rows (real + synthetic)
CTGAN_EPOCHS = 150              # CTGAN training epochs - higher = better fidelity, slower

# --- Stage 3: SMOTEENN cleanup ---
# Only classes still below this count after CTGAN get SMOTEENN oversampling.
# Classes at/above this are left completely alone - this is what keeps
# runtime realistic (no nearest-neighbor search across 2M+ untouched rows).
SMOTEENN_TARGET_PER_CLASS = 20_000


def log_and_print(report_lines, msg):
    print(msg)
    report_lines.append(msg)


# ============================================================
# 2. STAGE 1 - DOWNSAMPLE BENIGN (chunked, memory-safe)
# ============================================================

def downsample_benign(report_lines):
    """
    Reads golden12_features.csv in chunks. For each chunk:
      - keeps every non-BENIGN row untouched
      - keeps only a random fraction of BENIGN rows, sized so that
        across the whole file we land close to BENIGN_DOWNSAMPLE_TARGET
    Writes the result to an intermediate file, never holding the full
    15M+ row dataset in memory at once.
    """
    log_and_print(report_lines, "\n--- STAGE 1: Downsampling BENIGN ---")

    # First pass: just count total BENIGN rows (cheap - one column only)
    total_benign = 0
    total_rows = 0
    for chunk in pd.read_csv(INPUT_CSV, usecols=["unified_label"], chunksize=CHUNK_SIZE):
        total_rows += len(chunk)
        total_benign += int((chunk["unified_label"] == "BENIGN").sum())

    sample_fraction = min(1.0, BENIGN_DOWNSAMPLE_TARGET / total_benign)
    log_and_print(report_lines, f"Total rows: {total_rows:,} | Total BENIGN: {total_benign:,}")
    log_and_print(report_lines, f"BENIGN sample fraction: {sample_fraction:.5f} (target ~{BENIGN_DOWNSAMPLE_TARGET:,})")

    rng = np.random.default_rng(RANDOM_SEED)
    stage1_path = INPUT_CSV.replace(".csv", "_stage1_benign_downsampled.csv")
    if os.path.exists(stage1_path):
        os.remove(stage1_path)

    first_chunk = True
    kept_benign = 0
    kept_other = 0

    for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE, low_memory=False):
        is_benign = chunk["unified_label"] == "BENIGN"
        benign_rows = chunk[is_benign]
        other_rows = chunk[~is_benign]

        if len(benign_rows) > 0:
            keep_mask = rng.random(len(benign_rows)) < sample_fraction
            benign_rows = benign_rows[keep_mask]

        combined = pd.concat([other_rows, benign_rows], ignore_index=True)
        combined.to_csv(stage1_path, mode="a", index=False, header=first_chunk)
        first_chunk = False

        kept_benign += len(benign_rows)
        kept_other += len(other_rows)

    log_and_print(report_lines, f"BENIGN kept: {kept_benign:,} | All other classes kept in full: {kept_other:,}")
    return stage1_path


# ============================================================
# 3. STAGE 2 - CTGAN SYNTHESIS FOR EXTREME MINORITY CLASSES
# ============================================================

def ctgan_augment(stage1_path, report_lines):
    """
    Loads the full stage1 file (now small enough - BENIGN downsampled),
    identifies classes below CTGAN_THRESHOLD, trains a separate CTGAN
    per qualifying class on its REAL rows only, and generates enough
    synthetic rows to reach CTGAN_TARGET_PER_CLASS.
    """
    from ctgan import CTGAN

    log_and_print(report_lines, "\n--- STAGE 2: CTGAN synthesis for extreme minority classes ---")

    df = pd.read_csv(stage1_path, low_memory=False)
    class_counts = df["unified_label"].value_counts()

    qualifying_classes = [cls for cls, cnt in class_counts.items() if cnt < CTGAN_THRESHOLD]
    log_and_print(report_lines, f"Classes below {CTGAN_THRESHOLD:,} rows (CTGAN candidates): {qualifying_classes}")

    synthetic_frames = []
    for cls in qualifying_classes:
        real_rows = df[df["unified_label"] == cls][GOLDEN_12].reset_index(drop=True)
        n_real = len(real_rows)
        n_to_generate = max(0, CTGAN_TARGET_PER_CLASS - n_real)

        log_and_print(report_lines, f"\n  Class '{cls}': {n_real:,} real rows -> generating {n_to_generate:,} synthetic rows")

        if n_to_generate == 0:
            continue

        model = CTGAN(epochs=CTGAN_EPOCHS, verbose=False)
        model.fit(real_rows)  # all 12 Golden features are continuous - no discrete_columns needed

        synthetic = model.sample(n_to_generate)
        synthetic["unified_label"] = cls
        synthetic["source_year"] = "synthetic_ctgan"
        synthetic_frames.append(synthetic)

    if synthetic_frames:
        all_synthetic = pd.concat(synthetic_frames, ignore_index=True)
        log_and_print(report_lines, f"\nTotal synthetic rows generated: {len(all_synthetic):,}")
    else:
        all_synthetic = pd.DataFrame(columns=df.columns)
        log_and_print(report_lines, "\nNo classes qualified for CTGAN synthesis.")

    combined = pd.concat([df, all_synthetic], ignore_index=True)
    return combined


# ============================================================
# 4. STAGE 3 - TARGETED SMOTEENN CLEANUP
# ============================================================

def smoteenn_cleanup(df, report_lines):
    """
    Two independent jobs, run in the correct order:

      1. SMOTE oversampling - ONLY for classes still below
         SMOTEENN_TARGET_PER_CLASS after the CTGAN stage. Classes
         already at/above target are left alone (sampling_strategy
         is explicit, not 'auto').

      2. ENN (Edited Nearest Neighbours) cleaning - runs ALWAYS,
         across the FULL combined dataset (real + CTGAN-synthetic +
         any SMOTE output), regardless of whether step 1 did
         anything. This is the part that removes noisy/ambiguous
         rows sitting on class boundaries - including synthetic
         rows CTGAN may have placed in an unrealistic spot. Skipping
         this just because no class needed MORE rows would silently
         drop the actual "cleanup" half of SMOTEENN.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import EditedNearestNeighbours

    log_and_print(report_lines, "\n--- STAGE 3: SMOTE oversampling (targeted) + ENN cleaning (always runs) ---")

    class_counts = df["unified_label"].value_counts().to_dict()
    log_and_print(report_lines, f"Class counts entering Stage 3: {class_counts}")

    X = df[GOLDEN_12].values
    y = df["unified_label"].values

    # --- Step 1: targeted SMOTE oversampling, only if any class still needs it ---
    sampling_strategy = {
        cls: SMOTEENN_TARGET_PER_CLASS
        for cls, cnt in class_counts.items()
        if cnt < SMOTEENN_TARGET_PER_CLASS
    }

    if sampling_strategy:
        log_and_print(report_lines, f"SMOTE will oversample: {sampling_strategy}")
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_SEED)
        X, y = smote.fit_resample(X, y)
        log_and_print(report_lines, f"Row count after SMOTE oversampling: {len(X):,}")
    else:
        log_and_print(report_lines, "No class below target - SMOTE oversampling step skipped "
                                      "(nothing to add), but ENN cleaning still runs below.")

    # --- Step 2: ENN cleaning - ALWAYS runs, across every class ---
    log_and_print(report_lines, "\nRunning ENN cleaning across the full combined dataset "
                                  "(removes noisy/ambiguous rows near class boundaries)...")
    enn = EditedNearestNeighbours(n_neighbors=3, n_jobs=-1)
    X_clean, y_clean = enn.fit_resample(X, y)

    result = pd.DataFrame(X_clean, columns=GOLDEN_12)
    result["unified_label"] = y_clean
    result["source_year"] = "post_smoteenn"

    rows_before_enn = len(X)
    rows_after_enn = len(X_clean)
    log_and_print(report_lines, f"Row count before ENN cleaning: {rows_before_enn:,}")
    log_and_print(report_lines, f"Row count after ENN cleaning:  {rows_after_enn:,}")
    log_and_print(report_lines, f"Rows removed as noisy/ambiguous: {rows_before_enn - rows_after_enn:,}")
    log_and_print(report_lines, "\nPer-class counts after ENN cleaning:")
    log_and_print(report_lines, str(result["unified_label"].value_counts()))

    return result


# ============================================================
# 5. MAIN
# ============================================================

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_lines = ["ADMO THESIS - CTGAN + SMOTEENN AUGMENTATION REPORT"]

    stage1_path = downsample_benign(report_lines)
    stage2_df = ctgan_augment(stage1_path, report_lines)

    log_and_print(report_lines, "\nClass counts after CTGAN stage:")
    log_and_print(report_lines, str(stage2_df["unified_label"].value_counts()))

    final_df = smoteenn_cleanup(stage2_df, report_lines)

    final_df.to_csv(OUTPUT_CSV, index=False)

    log_and_print(report_lines, "\n--- FINAL CLASS DISTRIBUTION ---")
    log_and_print(report_lines, str(final_df["unified_label"].value_counts()))
    log_and_print(report_lines, f"\nOutput written to: {OUTPUT_CSV}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
