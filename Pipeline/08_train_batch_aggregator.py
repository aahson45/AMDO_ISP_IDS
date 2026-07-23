"""
08_train_batch_aggregator.py
-------------------------------
Phase 2, Step 8: Tier 2 - Flow Batch Aggregator (5-minute window).

IMPORTANT DESIGN NOTE (read this before anything else):
Our canonical merged dataset (balanced_dataset.csv) has NO source-IP
column - it was deliberately dropped in 02_merge_and_clean.py because
2017's data never had one at all, and keeping it would have broken
schema alignment across years. That was the right call for building
a clean historical TRAINING set, but it means true per-source
aggregation cannot be done on that file.

Instead of re-running the expensive 19M-row merge, this script goes
back to the ONE raw file that genuinely has both Src IP and Timestamp:
02-20-2018.csv (confirmed in 00_diagnose.py - every other file in
both datasets lacks these fields entirely). This lets us build and
demonstrate REAL per-source aggregation logic on real historical
data, without fabricating source IDs or redoing the full pipeline.

Per the architecture diagram, Tier 2 has no "(XGBoost/...)" label
like Tier 1 and Tier 3 do - it's a FEATURE ENGINEERING step, not a
separately trained model. This script's output (the 16-D per-source,
per-window vectors) is meant to feed INTO Tier 3, not stand alone as
its own classifier.

Full production validation of Tier 2 happens in Phase 5, using live
lab-generated traffic captured via CICFlowMeter - where every single
captured flow naturally includes a real source IP, since it's live
capture rather than an anonymized public dataset.

Input:  raw/2018/02-20-2018.csv (the ONE file with Src IP + Timestamp)
Output: tier2_aggregated_windows.csv (16-D vectors, one per source
        per 5-minute window) + a report
"""

import pandas as pd
import numpy as np
import os
import json

# ============================================================
# 1. CONFIG
# ============================================================
RAW_FILE = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\raw\2018\02-20-2018.csv"
MAPPING_JSON = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\label_mapping.json"

OUTPUT_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\processed"
REPORT_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\report"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "tier2_aggregated_windows.csv")
OUTPUT_REPORT = os.path.join(REPORT_DIR, "08_batch_aggregator_report.txt")

CHUNK_SIZE = 200_000
WINDOW_SIZE = "5min"

# Golden 12, in this file's RAW 2018 abbreviated column names
# (we're reading directly from the raw file, not the renamed merged one)
GOLDEN_12_RAW_2018 = {
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd PSH Flags": "Fwd PSH Flags",
    "ACK Flag Cnt": "ACK Flag Count",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Flow Duration": "Flow Duration",
    "Flow Pkts/s": "Flow Packets/s",
    "Flow Byts/s": "Flow Bytes/s",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Flow IAT Mean": "Flow IAT Mean",
    "Fwd IAT Max": "Fwd IAT Max",
    "Bwd IAT Min": "Bwd IAT Min",
}
GOLDEN_12_CANONICAL = list(GOLDEN_12_RAW_2018.values())

DROP_SENTINEL = "DROP_DUPLICATE_HEADER_ROW"


def normalize_key(raw_label):
    """Identical logic to 01/02 - kept in sync manually."""
    if not isinstance(raw_label, str):
        return str(raw_label)
    cleaned = raw_label.strip().lower()
    cleaned = cleaned.replace("\u00ef\u00bf\u00bd", " ")
    cleaned = cleaned.replace("-", " ")
    cleaned = " ".join(cleaned.split())
    if cleaned == "label":
        return DROP_SENTINEL
    return cleaned


def shannon_entropy(series):
    """
    Computes Shannon entropy of a categorical column's value
    distribution: -sum(p * log2(p)) for each unique value's
    proportion p. Higher entropy = more diverse/spread-out values
    (e.g. a source hitting many different destination ports);
    lower entropy = concentrated on few values (e.g. always the
    same port). This is exactly the signal a port-scan or DDoS
    source tends to produce differently from normal traffic.
    """
    counts = series.value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_lines = ["ADMO THESIS - TIER 2 BATCH AGGREGATOR REPORT"]
    report_lines.append(f"Source file: {os.path.basename(RAW_FILE)} (the only file with Src IP + Timestamp)")

    with open(MAPPING_JSON, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    label_map_2018 = mapping_data["2018"]

    # --- Load the raw file, only the columns we actually need ---
    needed_cols = ["Src IP", "Timestamp", "Label"] + list(GOLDEN_12_RAW_2018.keys())
    df = pd.read_csv(RAW_FILE, usecols=needed_cols, encoding="ascii", low_memory=False)
    report_lines.append(f"\nRaw rows loaded: {len(df):,}")

    # --- Apply label mapping, drop duplicate-header rows ---
    df["unified_label"] = df["Label"].apply(
        lambda v: label_map_2018.get(normalize_key(v), None)
        if normalize_key(v) != DROP_SENTINEL else DROP_SENTINEL
    )
    df = df[df["unified_label"] != DROP_SENTINEL]
    df = df[df["unified_label"].notna()]
    report_lines.append(f"Rows after label mapping / dropping header artifacts: {len(df):,}")

    # --- Rename Golden 12 to canonical names ---
    df = df.rename(columns=GOLDEN_12_RAW_2018)

    # --- Handle Infinity/NaN in the rate columns (same known issue as 03) ---
    for col in ["Flow Packets/s", "Flow Bytes/s"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    rows_before_dropna = len(df)
    df = df.dropna(subset=GOLDEN_12_CANONICAL)
    report_lines.append(f"Rows dropped for NaN/Inf in Golden 12: {rows_before_dropna - len(df):,}")

    # --- Parse Timestamp ---
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
    rows_before_ts = len(df)
    df = df.dropna(subset=["Timestamp"])
    report_lines.append(f"Rows dropped for unparseable Timestamp: {rows_before_ts - len(df):,}")

    # --- Build the 5-minute window bucket ---
    df["window_start"] = df["Timestamp"].dt.floor(WINDOW_SIZE)

    report_lines.append(f"\nUnique source IPs: {df['Src IP'].nunique():,}")
    report_lines.append(f"Unique 5-minute windows: {df['window_start'].nunique():,}")

    # --- Group by (Src IP, window) and compute the 16-D vector ---
    remaining_11 = [c for c in GOLDEN_12_CANONICAL if c != "Flow Duration"]

    agg_rows = []
    group_cols = ["Src IP", "window_start"]
    for (src_ip, window), group in df.groupby(group_cols, sort=False):
        row = {
            "Src IP": src_ip,
            "window_start": window,
            "Flow_Count": len(group),
            "Mean_Flow_Duration": group["Flow Duration"].mean(),
            "Protocol_Entropy": shannon_entropy(group["Protocol"]) if "Protocol" in group.columns else np.nan,
            "Destination_Port_Entropy": shannon_entropy(group["Dst Port"]) if "Dst Port" in group.columns else np.nan,
        }
        for feat in remaining_11:
            row[f"Mean_{feat}"] = group[feat].mean()
        row["Unique_Destination_Ports"] = group["Dst Port"].nunique() if "Dst Port" in group.columns else np.nan
        # Window label: "any-attack-present" precedence, NOT majority vote.
        # Majority vote (.mode()) would label a window BENIGN the instant
        # attack flows are outnumbered by ordinary traffic from the same
        # source in the same window - which, empirically, erased all but
        # a tiny fraction of real attack windows (a well-documented
        # dilution effect in flow-aggregation IDS literature). Instead:
        # if ANY non-BENIGN flow exists in the window, the window is
        # labeled by whichever non-BENIGN class is most frequent among
        # the non-BENIGN flows specifically - only windows with zero
        # attack flows at all are labeled BENIGN.
        labels_in_group = group["unified_label"]
        non_benign = labels_in_group[labels_in_group != "BENIGN"]
        if len(non_benign) > 0:
            row["window_label"] = non_benign.mode().iloc[0]
        else:
            row["window_label"] = "BENIGN"
        row["Attack_Flow_Fraction"] = len(non_benign) / len(group)
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)

    report_lines.append(f"\nTotal (Src IP, window) groups produced: {len(agg_df):,}")
    feature_cols = [c for c in agg_df.columns
                     if c not in ["Src IP", "window_start", "window_label", "Attack_Flow_Fraction"]]
    report_lines.append(f"16-D vector columns produced: {len(feature_cols)}")
    report_lines.append("(Attack_Flow_Fraction is a diagnostic label-quality field, not one of the 16 "
                          "declared model input features - kept separately so mixed windows are visible.)")
    report_lines.append(f"\nWindow label distribution:\n{agg_df['window_label'].value_counts().to_string()}")

    agg_df.to_csv(OUTPUT_CSV, index=False)
    report_lines.append(f"\nOutput written to: {OUTPUT_CSV}")
    report_lines.append("\nREMINDER: this demonstrates Tier 2's aggregation logic on ONE real historical file "
                          "(02-20-2018.csv), the only one with genuine Src IP + Timestamp. Full production "
                          "validation of Tier 2 happens in Phase 5 with live lab-generated traffic, where "
                          "every captured flow naturally includes a real source IP.")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nReport saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
