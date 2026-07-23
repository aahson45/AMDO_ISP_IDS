"""
00_diagnose.py
--------------
Phase 2, Step 0: Inspect both raw datasets BEFORE any cleaning/merging.
This script does NOT modify any files. It only reads and reports.

Goal: know exactly what we're working with (column names, encodings,
row counts, structural inconsistencies) so later scripts don't silently
break on surprises.
"""

import pandas as pd
import os
import chardet

# ============================================================
# 1. CONFIG - your actual folder paths
# ============================================================
DIR_2017 = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\raw\2017"
DIR_2018 = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\raw\2018"

OUTPUT_REPORT = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\00_diagnose_report.txt"

CHUNK_SIZE = 200_000  # rows per chunk when reading large files

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def detect_encoding(filepath, sample_size=200_000):
    """
    Reads a small raw byte sample from the file and asks chardet
    to guess the text encoding. This is how we catch files that
    aren't plain UTF-8 (e.g., 2017's cp1252 en-dash issue) before
    pandas throws a confusing UnicodeDecodeError later.
    """
    with open(filepath, "rb") as f:
        raw = f.read(sample_size)
    result = chardet.detect(raw)
    return result  # dict with 'encoding' and 'confidence'


def count_rows_chunked(filepath, encoding):
    """
    Counts total rows without loading the whole file into memory at once.
    Necessary for the ~4GB 02-20-2018.csv file specifically.
    """
    total = 0
    try:
        for chunk in pd.read_csv(filepath, encoding=encoding, chunksize=CHUNK_SIZE,
                                   low_memory=False, on_bad_lines="skip"):
            total += len(chunk)
    except Exception as e:
        return f"ERROR while counting rows: {e}"
    return total


def get_columns(filepath, encoding):
    """
    Reads ONLY the header row - fast, no need to load full file
    just to see column names.
    """
    try:
        df_head = pd.read_csv(filepath, encoding=encoding, nrows=0)
        return list(df_head.columns)
    except Exception as e:
        return f"ERROR while reading columns: {e}"


def report_file(filepath, report_lines):
    """
    Runs the full diagnostic on one file and appends results
    to the report_lines list (so we can write everything to
    one text file at the end).
    """
    filename = os.path.basename(filepath)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)

    report_lines.append(f"\n{'='*70}")
    report_lines.append(f"FILE: {filename}")
    report_lines.append(f"Size: {size_mb:,.2f} MB")

    # --- Encoding detection ---
    enc_result = detect_encoding(filepath)
    detected_encoding = enc_result["encoding"]
    confidence = enc_result["confidence"]
    report_lines.append(f"Detected encoding: {detected_encoding} (confidence: {confidence:.2f})")

    # Fallback: if detection is low-confidence or None, default to utf-8,
    # but flag it clearly so we know to double check manually.
    encoding_to_use = detected_encoding if detected_encoding else "utf-8"

    # --- Columns ---
    columns = get_columns(filepath, encoding_to_use)
    if isinstance(columns, str):  # means an error string was returned
        report_lines.append(f"COLUMN READ ERROR: {columns}")
        # Try again with latin1 as a fallback, since that's the known
        # fix for the 2017 encoding issue noted in project history
        columns_fallback = get_columns(filepath, "latin1")
        report_lines.append(f"Retried with latin1: {columns_fallback if isinstance(columns_fallback, list) else columns_fallback}")
        columns = columns_fallback if isinstance(columns_fallback, list) else []
        encoding_to_use = "latin1"

    if isinstance(columns, list):
        report_lines.append(f"Number of columns: {len(columns)}")
        # Flag any column names with leading/trailing whitespace
        whitespace_flags = [c for c in columns if c != c.strip()]
        if whitespace_flags:
            report_lines.append(f"COLUMNS WITH LEADING/TRAILING WHITESPACE: {whitespace_flags}")
        else:
            report_lines.append("No whitespace issues in column names.")
        report_lines.append(f"Columns: {columns}")

    # --- Row count ---
    row_count = count_rows_chunked(filepath, encoding_to_use)
    report_lines.append(f"Row count (excluding header): {row_count:,}" if isinstance(row_count, int) else row_count)

    return encoding_to_use, columns if isinstance(columns, list) else []


# ============================================================
# 3. MAIN DIAGNOSTIC RUN
# ============================================================

def main():
    report_lines = []
    report_lines.append("ADMO THESIS - RAW DATASET DIAGNOSTIC REPORT")
    report_lines.append(f"2017 folder: {DIR_2017}")
    report_lines.append(f"2018 folder: {DIR_2018}")

    # --- 2017 files ---
    report_lines.append("\n\n" + "#" * 70)
    report_lines.append("# CICIDS-2017 FILES")
    report_lines.append("#" * 70)

    files_2017 = [f for f in os.listdir(DIR_2017) if f.endswith(".csv")]
    columns_2017_by_file = {}

    for fname in sorted(files_2017):
        full_path = os.path.join(DIR_2017, fname)
        encoding_used, cols = report_file(full_path, report_lines)
        columns_2017_by_file[fname] = cols

    # Check if all 2017 files share identical columns
    report_lines.append(f"\n{'-'*70}")
    report_lines.append("2017 CROSS-FILE COLUMN CONSISTENCY CHECK:")
    unique_column_sets = set(tuple(cols) for cols in columns_2017_by_file.values())
    if len(unique_column_sets) == 1:
        report_lines.append("All 2017 files share IDENTICAL column structure. Good.")
    else:
        report_lines.append(f"WARNING: {len(unique_column_sets)} different column structures found across 2017 files!")
        for fname, cols in columns_2017_by_file.items():
            report_lines.append(f"  {fname}: {len(cols)} columns")

    # --- 2018 files ---
    report_lines.append("\n\n" + "#" * 70)
    report_lines.append("# CSE-CIC-IDS2018 FILES")
    report_lines.append("#" * 70)

    files_2018 = [f for f in os.listdir(DIR_2018) if f.endswith(".csv")]
    columns_2018_by_file = {}

    for fname in sorted(files_2018):
        full_path = os.path.join(DIR_2018, fname)
        encoding_used, cols = report_file(full_path, report_lines)
        columns_2018_by_file[fname] = cols

    # Check if all 2018 files share identical columns
    # (This is a KNOWN issue for this dataset - some days have extra/missing
    # or reordered columns, which is why we check explicitly here.)
    report_lines.append(f"\n{'-'*70}")
    report_lines.append("2018 CROSS-FILE COLUMN CONSISTENCY CHECK:")
    unique_column_sets_2018 = set(tuple(cols) for cols in columns_2018_by_file.values())
    if len(unique_column_sets_2018) == 1:
        report_lines.append("All 2018 files share IDENTICAL column structure. Good.")
    else:
        report_lines.append(f"WARNING: {len(unique_column_sets_2018)} different column structures found across 2018 files!")
        for fname, cols in columns_2018_by_file.items():
            report_lines.append(f"  {fname}: {len(cols)} columns")
        # Show exactly which columns differ, using the first file as baseline
        baseline_name = sorted(columns_2018_by_file.keys())[0]
        baseline_cols = set(columns_2018_by_file[baseline_name])
        report_lines.append(f"\nUsing '{baseline_name}' as baseline for diff:")
        for fname, cols in columns_2018_by_file.items():
            cols_set = set(cols)
            missing = baseline_cols - cols_set
            extra = cols_set - baseline_cols
            if missing or extra:
                report_lines.append(f"  {fname}: missing={missing}, extra={extra}")

    # ============================================================
    # 4. WRITE REPORT TO FILE + PRINT TO CONSOLE
    # ============================================================
    report_text = "\n".join(str(line) for line in report_lines)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n\nFull report also saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
