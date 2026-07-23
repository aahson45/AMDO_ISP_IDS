"""
01_label_normalization.py
--------------------------
Phase 2, Step 1: Build and validate the label taxonomy mapping.

This script does NOT merge or rewrite the big CSVs yet (that's 02's job).
Its job is narrower and safer: read ONLY the Label column from every file
(fast - we don't load all 80 columns), collect every unique raw label
value that actually exists in the data, map it onto the unified 6-class
taxonomy, and loudly flag anything that doesn't match - so we catch
surprises now, not silently mid-merge.

Output: a text report + a reusable mapping dictionary saved as JSON,
which 02_merge_and_clean.py will load and apply.
"""

import pandas as pd
import os
import json

# ============================================================
# 1. CONFIG
# ============================================================
DIR_2017 = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\raw\2017"
DIR_2018 = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\raw\2018"

OUTPUT_REPORT = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\01_label_report.txt"
OUTPUT_MAPPING_JSON = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\label_mapping.json"

CHUNK_SIZE = 200_000

# 2017 files use latin1 encoding (confirmed necessary by 00_diagnose.py -
# the Thursday WebAttacks file has a byte that breaks ascii/utf-8 decoding
# partway through, in a Label value).
ENCODING_2017 = "latin1"

# 2018 files were confirmed as clean ascii by 00_diagnose.py.
ENCODING_2018 = "ascii"

# 02-20-2018.csv has 4 extra identifier columns (Flow ID, Src IP, Src Port,
# Dst IP) that no other file has. We don't need them for label scanning
# (we only read the Label column anyway), but this is noted here since
# 02_merge_and_clean.py will need to drop them before concatenation.
FILE_WITH_EXTRA_COLS = "02-20-2018.csv"

# ============================================================
# 2. THE UNIFIED 6-CLASS TAXONOMY MAPPING
# ============================================================
# These mappings are built from the well-documented CICIDS-2017 and
# CSE-CIC-IDS2018 label sets. Keys are lowercased and stripped before
# matching, so casing/whitespace differences in the raw data don't
# cause false "unmapped" flags.

LABEL_MAP_2017 = {
    "benign": "BENIGN",

    "dos hulk": "DoS_DDoS",
    "dos goldeneye": "DoS_DDoS",
    "dos slowloris": "DoS_DDoS",
    "dos slowhttptest": "DoS_DDoS",
    "ddos": "DoS_DDoS",
    "heartbleed": "DoS_DDoS",

    "ftp patator": "BRUTE_FORCE",
    "ssh patator": "BRUTE_FORCE",

    "web attack brute force": "WEB_ATTACK",
    "web attack xss": "WEB_ATTACK",
    "web attack sql injection": "WEB_ATTACK",

    "bot": "BOTNET",

    "infiltration": "INFILTRATION",

    "portscan": "INFILTRATION",  # flagged in project notes as potential 7th RECON class
}

LABEL_MAP_2018 = {
    "benign": "BENIGN",

    "dos attacks hulk": "DoS_DDoS",
    "dos attacks goldeneye": "DoS_DDoS",
    "dos attacks slowloris": "DoS_DDoS",
    "dos attacks slowhttptest": "DoS_DDoS",
    "ddos attacks loic http": "DoS_DDoS",
    "ddos attack hoic": "DoS_DDoS",
    "ddos attack loic udp": "DoS_DDoS",

    "ftp bruteforce": "BRUTE_FORCE",
    "ssh bruteforce": "BRUTE_FORCE",

    "brute force web": "WEB_ATTACK",
    "brute force xss": "WEB_ATTACK",
    "sql injection": "WEB_ATTACK",

    "bot": "BOTNET",

    "infilteration": "INFILTRATION",  # official dataset's own typo, kept as-is for matching
}

# Sentinel: rows where the Label column literally contains the word "label"
# are duplicated header rows accidentally embedded in the CIC-IDS2018 data
# (a known artifact of how the dataset was assembled), NOT real flow
# records. These must be DROPPED entirely in 02_merge_and_clean.py.
DROP_SENTINEL = "DROP_DUPLICATE_HEADER_ROW"


def normalize_key(raw_label):
    """
    Cleans a raw label value so it can be reliably matched against the
    mapping dictionaries above, regardless of stray whitespace, casing,
    hyphen placement, or mangled special characters. The en-dash in
    2017's Web Attack labels was corrupted upstream into the UTF-8 bytes
    for U+FFFD, which show up as the 3-character sequence 'ï¿½' once
    misread as latin1 - we specifically strip that exact sequence.
    """
    if not isinstance(raw_label, str):
        return str(raw_label)
    cleaned = raw_label.strip().lower()

    # Strip the specific mangled byte sequence (UTF-8 U+FFFD misread as latin1)
    cleaned = cleaned.replace("\u00ef\u00bf\u00bd", " ")

    # Replace ALL hyphens with spaces - labels are inconsistent about hyphen
    # placement across both datasets ("FTP-Patator" vs "DoS attacks-Hulk"
    # vs "Brute Force -Web" vs "DDOS attack-LOIC-UDP"). Dictionary keys
    # above are written with NO hyphens (spaces only) to match this.
    cleaned = cleaned.replace("-", " ")

    # Collapse any run of whitespace down to a single space
    cleaned = " ".join(cleaned.split())

    if cleaned == "label":
        return DROP_SENTINEL

    return cleaned


def scan_labels(filepath, encoding, report_lines, label_map):
    """
    Reads ONLY the Label column (fast - no need to load all 80 columns),
    counts every unique raw value, and reports how each maps (or fails
    to map) onto the unified taxonomy.
    """
    filename = os.path.basename(filepath)
    report_lines.append(f"\n{'-'*70}")
    report_lines.append(f"FILE: {filename}")

    # Find the exact label column name first (it varies: 'Label' vs ' Label')
    header = pd.read_csv(filepath, encoding=encoding, nrows=0)
    label_col_candidates = [c for c in header.columns if c.strip().lower() == "label"]
    if not label_col_candidates:
        report_lines.append("  ERROR: no Label column found!")
        return {}
    label_col = label_col_candidates[0]

    value_counts = {}
    try:
        for chunk in pd.read_csv(filepath, encoding=encoding, usecols=[label_col],
                                   chunksize=CHUNK_SIZE, low_memory=False, on_bad_lines="skip"):
            counts = chunk[label_col].value_counts()
            for val, cnt in counts.items():
                value_counts[val] = value_counts.get(val, 0) + int(cnt)
    except Exception as e:
        report_lines.append(f"  ERROR while scanning labels: {e}")
        return {}

    unmapped_found = False
    for raw_val, count in sorted(value_counts.items(), key=lambda x: -x[1]):
        key = normalize_key(raw_val)
        if key == DROP_SENTINEL:
            report_lines.append(f"  DROP (duplicate embedded header row): '{raw_val}' - count: {count:,}")
            continue
        mapped_class = label_map.get(key, None)
        if mapped_class is None:
            report_lines.append(f"  UNMAPPED: '{raw_val}' (normalized key: '{key}') - count: {count:,}")
            unmapped_found = True
        else:
            report_lines.append(f"  '{raw_val}' -> {mapped_class} (count: {count:,})")

    if not unmapped_found:
        report_lines.append("  All real labels in this file mapped successfully.")

    return value_counts


# ============================================================
# 3. MAIN RUN
# ============================================================

def main():
    report_lines = []
    report_lines.append("ADMO THESIS - LABEL NORMALIZATION REPORT")

    # --- 2017 ---
    report_lines.append("\n\n" + "#" * 70)
    report_lines.append("# CICIDS-2017 LABEL SCAN")
    report_lines.append("#" * 70)

    files_2017 = sorted(f for f in os.listdir(DIR_2017) if f.endswith(".csv"))
    all_raw_labels_2017 = {}
    for fname in files_2017:
        full_path = os.path.join(DIR_2017, fname)
        counts = scan_labels(full_path, ENCODING_2017, report_lines, LABEL_MAP_2017)
        for val, cnt in counts.items():
            all_raw_labels_2017[val] = all_raw_labels_2017.get(val, 0) + cnt

    # --- 2018 ---
    report_lines.append("\n\n" + "#" * 70)
    report_lines.append("# CSE-CIC-IDS2018 LABEL SCAN")
    report_lines.append("#" * 70)

    files_2018 = sorted(f for f in os.listdir(DIR_2018) if f.endswith(".csv"))
    all_raw_labels_2018 = {}
    for fname in files_2018:
        full_path = os.path.join(DIR_2018, fname)
        counts = scan_labels(full_path, ENCODING_2018, report_lines, LABEL_MAP_2018)
        for val, cnt in counts.items():
            all_raw_labels_2018[val] = all_raw_labels_2018.get(val, 0) + cnt

    # --- Overall summary: any unmapped labels across BOTH datasets ---
    report_lines.append("\n\n" + "#" * 70)
    report_lines.append("# OVERALL UNMAPPED LABEL SUMMARY (across all files)")
    report_lines.append("#" * 70)

    any_unmapped = False
    total_drop_rows = 0
    for year, all_labels, label_map in [("2017", all_raw_labels_2017, LABEL_MAP_2017),
                                          ("2018", all_raw_labels_2018, LABEL_MAP_2018)]:
        for raw_val, count in all_labels.items():
            key = normalize_key(raw_val)
            if key == DROP_SENTINEL:
                total_drop_rows += count
                continue
            if label_map.get(key) is None:
                report_lines.append(f"  [{year}] UNMAPPED: '{raw_val}' - total count: {count:,}")
                any_unmapped = True

    report_lines.append(f"\n  Total duplicate-header rows to drop (2018 artifact): {total_drop_rows:,}")

    if not any_unmapped:
        report_lines.append("  No genuinely unmapped labels found. Safe to proceed to 02_merge_and_clean.py.")
        report_lines.append("  (Remember: 02_merge_and_clean.py must explicitly DROP rows where the Label "
                              "column normalizes to 'label' - these are embedded duplicate header rows, not real data.)")
    else:
        report_lines.append("\n  ACTION NEEDED: add the unmapped labels above to LABEL_MAP_2017 or LABEL_MAP_2018 "
                              "in this script before proceeding, then re-run.")

    # --- Save the mapping dictionaries for 02_merge_and_clean.py to reuse ---
    combined_mapping = {
        "2017": LABEL_MAP_2017,
        "2018": LABEL_MAP_2018,
        "file_with_extra_cols_2018": FILE_WITH_EXTRA_COLS,
        "encoding_2017": ENCODING_2017,
        "encoding_2018": ENCODING_2018,
    }
    with open(OUTPUT_MAPPING_JSON, "w", encoding="utf-8") as f:
        json.dump(combined_mapping, f, indent=2)

    # --- Write report ---
    report_text = "\n".join(str(line) for line in report_lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n\nFull report saved to: {OUTPUT_REPORT}")
    print(f"Mapping dictionary saved to: {OUTPUT_MAPPING_JSON}")


if __name__ == "__main__":
    main()