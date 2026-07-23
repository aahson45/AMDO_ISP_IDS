"""
04_class_balance_report.py
----------------------------
Phase 2, Step 4: Report the class distribution AFTER feature selection
(03_feature_selection.py), and explicitly check whether the ~98.6K
rows dropped for NaN/Infinity in the Golden 12 features hit any
minority class harder than others - proportionally, not just in
raw count.

This matters because WEB_ATTACK only has 3,108 examples total; even
a small number of dropped rows there is a much bigger deal than the
same number dropped from 15.7M BENIGN rows.

Input:  golden12_features.csv (from 03_feature_selection.py)
Output: class_balance_report.txt + class_balance_table.csv,
        both written to the shared /report folder from now on.
"""

import pandas as pd
import os

# ============================================================
# 1. CONFIG
# ============================================================
INPUT_CSV = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\golden12_features.csv"

REPORT_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set\report"
OUTPUT_REPORT = os.path.join(REPORT_DIR, "04_class_balance_report.txt")
OUTPUT_TABLE_CSV = os.path.join(REPORT_DIR, "class_balance_table.csv")

CHUNK_SIZE = 200_000

# Baseline class counts BEFORE feature selection, taken directly from
# 02_merge_and_clean.py's own printed report - this is what we compare
# against to see what feature selection's NaN/Inf drop actually cost
# each class, proportionally.
BASELINE_COUNTS = {
    "BENIGN": 15_757_805,
    "DoS_DDoS": 2_298_932,
    "BRUTE_FORCE": 394_784,
    "INFILTRATION": 320_900,
    "BOTNET": 288_157,
    "WEB_ATTACK": 3_108,
}

# Any class losing more than this fraction of its own rows gets
# flagged explicitly - not a hard error, just a loud, visible warning
# worth checking before proceeding to augmentation (05).
PROPORTIONAL_LOSS_WARNING_THRESHOLD = 0.02  # 2%


# ============================================================
# 2. MAIN RUN
# ============================================================

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    report_lines = []
    report_lines.append("ADMO THESIS - CLASS BALANCE REPORT (post feature selection)")

    # --- Count classes in golden12_features.csv, chunked ---
    after_counts = {}
    total_rows = 0
    for chunk in pd.read_csv(INPUT_CSV, usecols=["unified_label"], chunksize=CHUNK_SIZE):
        total_rows += len(chunk)
        for cls, cnt in chunk["unified_label"].value_counts().items():
            after_counts[cls] = after_counts.get(cls, 0) + int(cnt)

    report_lines.append(f"\nTotal rows after feature selection: {total_rows:,}")

    # --- Build the comparison table ---
    table_rows = []
    majority_count_after = max(after_counts.values())

    report_lines.append(f"\n{'Class':<15}{'Before':>14}{'After':>14}{'Dropped':>12}{'% of class lost':>18}{'% of total (after)':>20}")
    report_lines.append("-" * 95)

    any_warning = False
    for cls in BASELINE_COUNTS:
        before = BASELINE_COUNTS[cls]
        after = after_counts.get(cls, 0)
        dropped = before - after
        pct_class_lost = (dropped / before * 100) if before > 0 else 0.0
        pct_of_total_after = (after / total_rows * 100) if total_rows > 0 else 0.0
        imbalance_ratio = majority_count_after / after if after > 0 else float("inf")

        flag = ""
        if pct_class_lost / 100 > PROPORTIONAL_LOSS_WARNING_THRESHOLD:
            flag = "  <-- WARNING: disproportionate loss"
            any_warning = True

        report_lines.append(
            f"{cls:<15}{before:>14,}{after:>14,}{dropped:>12,}{pct_class_lost:>17.3f}%{pct_of_total_after:>19.3f}%{flag}"
        )

        table_rows.append({
            "Class": cls,
            "Count_Before_Feature_Selection": before,
            "Count_After_Feature_Selection": after,
            "Rows_Dropped": dropped,
            "Pct_of_Class_Lost": round(pct_class_lost, 4),
            "Pct_of_Total_After": round(pct_of_total_after, 4),
            "Imbalance_Ratio_vs_Majority": round(imbalance_ratio, 1) if after > 0 else None,
        })

    report_lines.append("")
    if any_warning:
        report_lines.append("ACTION NEEDED: at least one class lost a disproportionate share of its "
                              "rows to NaN/Infinity filtering. Review before proceeding to 05_ctgan_smoteenn_augment.py - "
                              "a class already this rare can't absorb unexpected additional loss.")
    else:
        report_lines.append("No class lost a disproportionate share of rows (all under "
                              f"{PROPORTIONAL_LOSS_WARNING_THRESHOLD*100:.0f}%). Safe to proceed to "
                              "05_ctgan_smoteenn_augment.py.")

    report_lines.append(f"\nWorst-case imbalance ratio (majority vs. smallest class): "
                          f"{majority_count_after:,} : {after_counts.get('WEB_ATTACK', 0):,} "
                          f"(~{majority_count_after / after_counts.get('WEB_ATTACK', 1):,.0f} : 1)")

    # --- Write outputs ---
    report_text = "\n".join(report_lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(OUTPUT_TABLE_CSV, index=False)
    print(f"\nReport saved to: {OUTPUT_REPORT}")
    print(f"Table saved to: {OUTPUT_TABLE_CSV}")


if __name__ == "__main__":
    main()
