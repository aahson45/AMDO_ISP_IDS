"""
02_merge_and_clean.py
----------------------
Phase 2, Step 2: Merge CICIDS-2017 and CSE-CIC-IDS2018 into one clean,
unified CSV with consistent columns and a unified 6-class label.

Reuses the label mapping validated by 01_label_normalization.py
(loaded from label_mapping.json - no need to redefine it here).

Processes files in CHUNKS (not loaded fully into memory) since the
combined dataset is ~18M+ rows - this keeps memory usage flat
regardless of total size, safe even for the 4GB 02-20-2018.csv file.

Output: one combined CSV, written incrementally (each chunk appended),
never held fully in memory at once.
"""

import pandas as pd
import os
import json

# ============================================================
# 1. CONFIG
# ============================================================
DIR_2017 = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\raw\2017"
DIR_2018 = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\raw\2018"

MAPPING_JSON = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\label_mapping.json"
OUTPUT_CSV = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\combined_clean.csv"
OUTPUT_REPORT = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\02_merge_report.txt"

CHUNK_SIZE = 200_000

# Files needing special handling (found during diagnosis)
FILE_WITH_EXTRA_COLS_2018 = "02-20-2018.csv"
EXTRA_COLS_TO_DROP_2018 = ["Flow ID", "Src IP", "Src Port", "Dst IP"]

# Columns that exist in 2018's "baseline" schema but NOT anywhere in 2017 -
# dropped so both years end up with genuinely identical column sets,
# rather than silently producing NaN-filled columns for one year.
COLS_2018_NOT_IN_2017 = ["Protocol", "Timestamp"]

# 2017's original header row lists "Fwd Header Length" twice; pandas
# auto-renames the second occurrence to "Fwd Header Length.1". It's a
# duplicate of real data, not a distinct feature - drop it.
COLS_2017_DUPLICATE = ["Fwd Header Length.1"]

# 2018's abbreviated column names -> 2017's full column names
# (reused from the project's existing Label_Mapping.py)
COL_MAP_2018_TO_2017 = {
    "ACK Flag Cnt": "ACK Flag Count",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Header Len": "Bwd Header Length",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Dst Port": "Destination Port",
    "ECE Flag Cnt": "ECE Flag Count",
    "FIN Flag Cnt": "FIN Flag Count",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Header Len": "Fwd Header Length",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Fwd Seg Size Min": "min_seg_size_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "PSH Flag Cnt": "PSH Flag Count",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "Pkt Size Avg": "Average Packet Size",
    "RST Flag Cnt": "RST Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "URG Flag Cnt": "URG Flag Count",
}

DROP_SENTINEL = "DROP_DUPLICATE_HEADER_ROW"


# ============================================================
# 2. LOAD THE VALIDATED LABEL MAPPING (from 01_label_normalization.py)
# ============================================================
with open(MAPPING_JSON, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

LABEL_MAP_2017 = mapping_data["2017"]
LABEL_MAP_2018 = mapping_data["2018"]
ENCODING_2017 = mapping_data["encoding_2017"]
ENCODING_2018 = mapping_data["encoding_2018"]


def normalize_key(raw_label):
    """Identical logic to 01_label_normalization.py - kept in sync manually
    since these numbered scripts are designed to run standalone."""
    if not isinstance(raw_label, str):
        return str(raw_label)
    cleaned = raw_label.strip().lower()
    cleaned = cleaned.replace("\u00ef\u00bf\u00bd", " ")
    cleaned = cleaned.replace("-", " ")
    cleaned = " ".join(cleaned.split())
    if cleaned == "label":
        return DROP_SENTINEL
    return cleaned


# ============================================================
# 3. PER-CHUNK CLEANING FUNCTIONS
# ============================================================

def clean_chunk_2017(chunk):
    """
    Cleans one chunk of a 2017 file:
    - strips whitespace from column names
    - drops the known duplicate column
    - adds unified_label column, drops duplicate-header rows
    - adds source_year column
    """
    chunk.columns = [c.strip() for c in chunk.columns]
    chunk = chunk.drop(columns=[c for c in COLS_2017_DUPLICATE if c in chunk.columns])

    chunk["unified_label"] = chunk["Label"].apply(
        lambda v: LABEL_MAP_2017.get(normalize_key(v), None)
        if normalize_key(v) != DROP_SENTINEL else DROP_SENTINEL
    )
    chunk = chunk[chunk["unified_label"] != DROP_SENTINEL]
    chunk = chunk[chunk["unified_label"].notna()]  # safety: drop anything that still failed to map

    chunk["source_year"] = 2017
    return chunk


def clean_chunk_2018(chunk, filename):
    """
    Cleans one chunk of a 2018 file:
    - drops the extra identifier columns (only present in 02-20-2018.csv)
    - drops Protocol/Timestamp (not present in 2017 at all)
    - renames abbreviated columns to match 2017's naming
    - adds unified_label column, drops duplicate-header rows
    - adds source_year column
    """
    chunk.columns = [c.strip() for c in chunk.columns]

    if filename == FILE_WITH_EXTRA_COLS_2018:
        chunk = chunk.drop(columns=[c for c in EXTRA_COLS_TO_DROP_2018 if c in chunk.columns])

    chunk = chunk.drop(columns=[c for c in COLS_2018_NOT_IN_2017 if c in chunk.columns])
    chunk = chunk.rename(columns=COL_MAP_2018_TO_2017)

    chunk["unified_label"] = chunk["Label"].apply(
        lambda v: LABEL_MAP_2018.get(normalize_key(v), None)
        if normalize_key(v) != DROP_SENTINEL else DROP_SENTINEL
    )
    chunk = chunk[chunk["unified_label"] != DROP_SENTINEL]
    chunk = chunk[chunk["unified_label"].notna()]

    chunk["source_year"] = 2018
    return chunk


# ============================================================
# 4. MAIN MERGE RUN
# ============================================================

def main():
    report_lines = []
    report_lines.append("ADMO THESIS - MERGE & CLEAN REPORT")

    # Remove any old output file first, since we'll be APPENDING to it
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    first_chunk_written = False
    expected_columns = None
    class_counts = {}
    total_rows_written = 0
    total_rows_dropped = 0

    # --- Process 2017 files ---
    files_2017 = sorted(f for f in os.listdir(DIR_2017) if f.endswith(".csv"))
    for fname in files_2017:
        full_path = os.path.join(DIR_2017, fname)
        report_lines.append(f"\nProcessing 2017 file: {fname}")
        for raw_chunk in pd.read_csv(full_path, encoding=ENCODING_2017, chunksize=CHUNK_SIZE,
                                       low_memory=False, on_bad_lines="skip"):
            rows_before = len(raw_chunk)
            cleaned = clean_chunk_2017(raw_chunk)
            total_rows_dropped += (rows_before - len(cleaned))

            # --- Safety check: confirm column set matches what we expect ---
            if expected_columns is None:
                expected_columns = set(cleaned.columns)
            else:
                mismatch = set(cleaned.columns).symmetric_difference(expected_columns)
                if mismatch:
                    raise ValueError(
                        f"COLUMN MISMATCH detected in {fname}! "
                        f"Difference vs. expected: {mismatch}. "
                        f"Stopping before writing a broken merged file."
                    )

            cleaned.to_csv(OUTPUT_CSV, mode="a", index=False, header=not first_chunk_written)
            first_chunk_written = True
            total_rows_written += len(cleaned)

            for cls, cnt in cleaned["unified_label"].value_counts().items():
                class_counts[cls] = class_counts.get(cls, 0) + int(cnt)

    # --- Process 2018 files ---
    files_2018 = sorted(f for f in os.listdir(DIR_2018) if f.endswith(".csv"))
    for fname in files_2018:
        full_path = os.path.join(DIR_2018, fname)
        report_lines.append(f"\nProcessing 2018 file: {fname}")
        for raw_chunk in pd.read_csv(full_path, encoding=ENCODING_2018, chunksize=CHUNK_SIZE,
                                       low_memory=False, on_bad_lines="skip"):
            rows_before = len(raw_chunk)
            cleaned = clean_chunk_2018(raw_chunk, fname)
            total_rows_dropped += (rows_before - len(cleaned))

            mismatch = set(cleaned.columns).symmetric_difference(expected_columns)
            if mismatch:
                raise ValueError(
                    f"COLUMN MISMATCH detected in {fname}! "
                    f"Difference vs. expected (from 2017 baseline): {mismatch}. "
                    f"Stopping before writing a broken merged file."
                )

            cleaned.to_csv(OUTPUT_CSV, mode="a", index=False, header=not first_chunk_written)
            first_chunk_written = True
            total_rows_written += len(cleaned)

            for cls, cnt in cleaned["unified_label"].value_counts().items():
                class_counts[cls] = class_counts.get(cls, 0) + int(cnt)

    # --- Final report ---
    report_lines.append(f"\n\n{'='*70}")
    report_lines.append(f"MERGE COMPLETE")
    report_lines.append(f"Total rows written: {total_rows_written:,}")
    report_lines.append(f"Total rows dropped (unmapped/duplicate-header): {total_rows_dropped:,}")
    report_lines.append(f"\nFinal class distribution (unified 6-class taxonomy):")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {cls}: {cnt:,}")
    report_lines.append(f"\nOutput written to: {OUTPUT_CSV}")

    report_text = "\n".join(report_lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nFull report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
