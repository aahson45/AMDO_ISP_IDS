"""
03_feature_selection.py
-------------------------
Phase 2, Step 3: Reduce the cleaned, merged dataset down to the
"Golden 12" feature set - 12 header-level/timing-level features
chosen because they're extractable from packet headers and flow
timing alone (no payload inspection needed), making them:
  - encryption-agnostic (work on TLS/encrypted traffic)
  - SmartNIC-extractable (no deep packet inspection hardware needed)

Also handles a well-known CICIDS data quality issue: the rate-based
features (Flow Bytes/s, Flow Packets/s) can be Infinity or NaN
whenever Flow Duration is 0 (division by zero). We detect and report
this explicitly before deciding how to handle it - never silently.

Input:  combined_clean.csv (from 02_merge_and_clean.py)
Output: golden12_features.csv + a feature rationale table
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# 1. CONFIG
# ============================================================
INPUT_CSV = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\combined_clean.csv"
OUTPUT_CSV = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\golden12_features.csv"
OUTPUT_REPORT = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\03_feature_selection_report.txt"
OUTPUT_RATIONALE_CSV = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\golden12_rationale.csv"

CHUNK_SIZE = 200_000

# Golden 12 features, mapped from the thesis doc's abbreviated names
# to the actual column names present in combined_clean.csv (which uses
# 2017-style full names, our canonical post-merge schema).
GOLDEN_12 = [
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd PSH Flags",
    "ACK Flag Count",
    "Init_Win_bytes_forward",
    "Flow Duration",
    "Flow Packets/s",
    "Flow Bytes/s",
    "Fwd Packet Length Max",
    "Flow IAT Mean",
    "Fwd IAT Max",
    "Bwd IAT Min",
]

# Columns to carry through alongside the features themselves - needed
# by every later script (04 onward), not part of the ML feature set itself.
CARRY_THROUGH_COLS = ["unified_label", "source_year"]

# The two rate-based features known to produce Infinity when
# Flow Duration == 0 (a documented CICIDS data quality artifact).
RATE_COLUMNS_AT_RISK = ["Flow Packets/s", "Flow Bytes/s"]

# Rationale for each feature - why it was chosen, matching the
# "documented rationale table" required by the Phase 2 plan.
RATIONALE = {
    "Fwd Header Length":     ("Layer 4 Header", "Detects tunneling/exploit packets via abnormal forward header sizes."),
    "Bwd Header Length":     ("Layer 4 Header", "Detects backward-traffic asymmetry typical of C2/botnet beaconing."),
    "Fwd PSH Flags":         ("Layer 4 Header", "Flags rapid-fire push behavior seen in SSH/FTP brute-force attempts."),
    "ACK Flag Count":        ("Layer 4 Header", "High ACK counts are a signature of ACK-flood DDoS traffic."),
    "Init_Win_bytes_forward":("Layer 4 Header", "Captures window-size manipulation used in Slowloris-style DoS."),
    "Flow Duration":         ("Flow Metadata",  "Distinguishes fast volumetric attacks from slow-and-low DoS."),
    "Flow Packets/s":        ("Flow Metadata",  "Direct volumetric indicator - key DDoS signal."),
    "Flow Bytes/s":          ("Flow Metadata",  "Detects high-throughput exfiltration or flooding behavior."),
    "Fwd Packet Length Max": ("Flow Metadata",  "Large forward payloads correlate with web attacks (e.g. SQLi)."),
    "Flow IAT Mean":         ("Temporal",       "Captures regular timing intervals typical of automated/bot traffic."),
    "Fwd IAT Max":           ("Temporal",       "Flags long gaps characteristic of low-and-slow brute force timing."),
    "Bwd IAT Min":           ("Temporal",       "Very short backward intervals are typical of high-speed DDoS reflection."),
}


# ============================================================
# 2. MAIN RUN
# ============================================================

def main():
    report_lines = []
    report_lines.append("ADMO THESIS - FEATURE SELECTION REPORT (Golden 12)")

    # --- Step 1: confirm every Golden 12 column actually exists ---
    # (verify, don't assume - same discipline as every earlier script)
    header = pd.read_csv(INPUT_CSV, nrows=0)
    available_cols = set(header.columns)
    missing = [c for c in GOLDEN_12 if c not in available_cols]
    if missing:
        raise ValueError(
            f"MISSING EXPECTED COLUMNS: {missing}\n"
            f"Available columns in {INPUT_CSV}:\n{sorted(available_cols)}"
        )
    report_lines.append(f"\nAll 12 Golden features confirmed present in {os.path.basename(INPUT_CSV)}.")

    columns_to_load = GOLDEN_12 + CARRY_THROUGH_COLS

    # --- Step 2: process in chunks, tracking inf/NaN stats along the way ---
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    first_chunk_written = False
    total_rows_in = 0
    total_rows_out = 0
    inf_counts = {col: 0 for col in RATE_COLUMNS_AT_RISK}
    nan_counts_before = {col: 0 for col in GOLDEN_12}

    for chunk in pd.read_csv(INPUT_CSV, usecols=columns_to_load, chunksize=CHUNK_SIZE, low_memory=False):
        total_rows_in += len(chunk)

        # Count NaNs already present, before we touch anything
        for col in GOLDEN_12:
            nan_counts_before[col] += int(chunk[col].isna().sum())

        # Count and then handle Infinity in the rate columns
        for col in RATE_COLUMNS_AT_RISK:
            inf_mask = np.isinf(chunk[col])
            inf_counts[col] += int(inf_mask.sum())
            # Replace Infinity with NaN - this converts an unusable value
            # into a properly recognized missing value, which we then
            # drop below. We do NOT silently keep Infinity, since most
            # ML libraries (including XGBoost/LightGBM) will error or
            # behave unpredictably if Infinity reaches model training.
            chunk[col] = chunk[col].replace([np.inf, -np.inf], np.nan)

        # Drop any row where ANY Golden 12 feature ended up NaN
        # (whether it was already NaN, or became NaN from the Infinity fix)
        rows_before_drop = len(chunk)
        chunk = chunk.dropna(subset=GOLDEN_12)
        rows_dropped_this_chunk = rows_before_drop - len(chunk)

        chunk.to_csv(OUTPUT_CSV, mode="a", index=False, header=not first_chunk_written)
        first_chunk_written = True
        total_rows_out += len(chunk)

    # --- Step 3: report ---
    report_lines.append(f"\nTotal rows read:  {total_rows_in:,}")
    report_lines.append(f"Total rows kept:  {total_rows_out:,}")
    report_lines.append(f"Total rows dropped (NaN/Inf in a Golden 12 feature): {total_rows_in - total_rows_out:,}")

    report_lines.append(f"\nInfinity values found (before conversion to NaN):")
    for col, cnt in inf_counts.items():
        report_lines.append(f"  {col}: {cnt:,}")

    report_lines.append(f"\nNaN values already present (before any Infinity handling):")
    for col, cnt in nan_counts_before.items():
        report_lines.append(f"  {col}: {cnt:,}")

    report_lines.append(f"\nOutput written to: {OUTPUT_CSV}")

    report_text = "\n".join(report_lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)

    # --- Step 4: save the rationale table ---
    rationale_rows = []
    for feature in GOLDEN_12:
        category, reason = RATIONALE[feature]
        rationale_rows.append({"Feature": feature, "Category": category, "Rationale": reason})
    rationale_df = pd.DataFrame(rationale_rows)
    rationale_df.to_csv(OUTPUT_RATIONALE_CSV, index=False)
    print(f"\nRationale table saved to: {OUTPUT_RATIONALE_CSV}")


if __name__ == "__main__":
    main()
