"""
10_latency_bench.py
----------------------
Phase 2, Step 10: Type A latency benchmark.

This is PURE ML PIPELINE TIMING - no networking, no pfSense, no real
packets involved yet (that's Type B, Phase 3 onward). We're measuring
how long the ML code itself takes, using four markers:

  MP1 -----------> MP2 -----------> MP3 -----------> MP4
  (start)   [feature      [model        [decision
             assembly]     inference]    post-processing]

This breakdown matters more than one lump number - it tells us WHERE
time is actually spent, not just how much total time passes.

Tier 1 and Tier 3 are benchmarked SEPARATELY, since in the full
architecture they operate on different things at different timescales
(Tier 1: single flows, real-time; Tier 3: per-source aggregated
batches, every 5 minutes) - forcing them into one artificial
"cascade latency" number would misrepresent both.

Input:  models/tier1_fastpath_binary.json, models/tier3_full_classifier_xgboost.json,
        splits/test.csv (used only as a source of realistic feature vectors)
Output: latency_benchmark_raw.csv (every single timed iteration) +
        a summary report with mean/median/p95/p99/max per interval +
        a machine-readable JSON summary (for downstream slide/table
        generation, so real numbers never need to be hand-copied out
        of a .txt report)
"""

import json
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import os

# ============================================================
# 1. CONFIG
# ============================================================
BASE_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software"
TEST_CSV = os.path.join(BASE_DIR, "Data Set", "splits", "test.csv")
MODEL_DIR = os.path.join(BASE_DIR, "Data Set", "models")

TIER1_MODEL_PATH = os.path.join(MODEL_DIR, "tier1_fastpath_binary.json")
TIER3_MODEL_PATH = os.path.join(MODEL_DIR, "tier3_full_classifier_xgboost.json")

REPORT_DIR = os.path.join(BASE_DIR, "Data Set", "report")
RESULTS_DIR = os.path.join(BASE_DIR, "Data Set", "processed", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_REPORT = os.path.join(REPORT_DIR, "10_latency_bench_report.txt")
OUTPUT_RAW_CSV = os.path.join(REPORT_DIR, "latency_benchmark_raw.csv")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "latency_benchmark_summary.json")

GOLDEN_12 = [
    "Fwd Header Length", "Bwd Header Length", "Fwd PSH Flags", "ACK Flag Count",
    "Init_Win_bytes_forward", "Flow Duration", "Flow Packets/s", "Flow Bytes/s",
    "Fwd Packet Length Max", "Flow IAT Mean", "Fwd IAT Max", "Bwd IAT Min",
]
LABEL_COL = "unified_label"

N_WARMUP = 50
N_TIMED = 2000
RANDOM_SEED = 42

TIER1_TARGET_MS = 0.5
CLASS_NAMES = ['BENIGN', 'BOTNET', 'BRUTE_FORCE', 'DoS_DDoS', 'INFILTRATION', 'WEB_ATTACK']


def benchmark_tier1(model, sample_rows, report_lines):
    timings = []
    threshold = 0.85

    for i in range(N_WARMUP):
        row = sample_rows.iloc[[i % len(sample_rows)]][GOLDEN_12]
        _ = model.predict_proba(row)

    for i in range(N_TIMED):
        row_data = sample_rows.iloc[i % len(sample_rows)].to_dict()

        mp1 = time.perf_counter()
        row_df = pd.DataFrame([row_data])[GOLDEN_12]
        mp2 = time.perf_counter()

        proba = model.predict_proba(row_df)[0, 1]
        mp3 = time.perf_counter()

        decision = "THREAT" if proba >= threshold else "BENIGN"
        mp4 = time.perf_counter()

        timings.append({
            "tier": "Tier1",
            "feature_assembly_ms": (mp2 - mp1) * 1000,
            "inference_ms": (mp3 - mp2) * 1000,
            "postprocessing_ms": (mp4 - mp3) * 1000,
            "total_ms": (mp4 - mp1) * 1000,
        })

    return pd.DataFrame(timings)


def benchmark_tier3(model, sample_rows, report_lines):
    timings = []

    for i in range(N_WARMUP):
        row = sample_rows.iloc[[i % len(sample_rows)]][GOLDEN_12]
        _ = model.predict_proba(row)

    for i in range(N_TIMED):
        row_data = sample_rows.iloc[i % len(sample_rows)].to_dict()

        mp1 = time.perf_counter()
        row_df = pd.DataFrame([row_data])[GOLDEN_12]
        mp2 = time.perf_counter()

        proba = model.predict_proba(row_df)[0]
        mp3 = time.perf_counter()

        predicted_idx = int(np.argmax(proba))
        predicted_class = CLASS_NAMES[predicted_idx]
        mp4 = time.perf_counter()

        timings.append({
            "tier": "Tier3",
            "feature_assembly_ms": (mp2 - mp1) * 1000,
            "inference_ms": (mp3 - mp2) * 1000,
            "postprocessing_ms": (mp4 - mp3) * 1000,
            "total_ms": (mp4 - mp1) * 1000,
        })

    return pd.DataFrame(timings)


def summarize(df, tier_name, report_lines, target_ms=None):
    report_lines.append(f"\n--- {tier_name} latency summary (ms), n={len(df)} timed iterations ---")
    col_stats = {}
    for col in ["feature_assembly_ms", "inference_ms", "postprocessing_ms", "total_ms"]:
        s = df[col]
        stats = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }
        col_stats[col] = stats
        report_lines.append(
            f"{col:<22} mean={stats['mean']:.4f}  median={stats['median']:.4f}  "
            f"p95={stats['p95']:.4f}  p99={stats['p99']:.4f}  max={stats['max']:.4f}"
        )

    target_info = None
    if target_ms is not None:
        mean_total = df["total_ms"].mean()
        p95_total = df["total_ms"].quantile(0.95)
        meets_mean = mean_total < target_ms
        meets_p95 = p95_total < target_ms
        report_lines.append(f"\nArchitecture target: < {target_ms} ms")
        report_lines.append(f"Mean total latency:  {mean_total:.4f} ms "
                              f"({'MEETS' if meets_mean else 'ABOVE'} target)")
        report_lines.append(f"P95 total latency:   {p95_total:.4f} ms "
                              f"({'MEETS' if meets_p95 else 'ABOVE'} target)")
        report_lines.append("NOTE: this is un-quantized (no INT8), pure-Python inference - "
                              "INT8 quantization and a compiled inference path are Phase 5 "
                              "deployment steps expected to close most of any remaining gap.")
        target_info = {
            "target_ms": target_ms,
            "mean_total_ms": float(mean_total),
            "p95_total_ms": float(p95_total),
            "meets_mean_target": bool(meets_mean),
            "meets_p95_target": bool(meets_p95),
        }

    return {"breakdown": col_stats, "target_comparison": target_info}


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_lines = ["ADMO THESIS - TYPE A LATENCY BENCHMARK (MP1-MP4)"]
    report_lines.append(f"Warmup calls: {N_WARMUP} | Timed iterations per tier: {N_TIMED}")

    test_df = pd.read_csv(TEST_CSV, low_memory=False)
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(len(test_df), size=min(500, len(test_df)), replace=False)
    sample_rows = test_df.iloc[sample_idx].reset_index(drop=True)
    report_lines.append(f"Sampled {len(sample_rows)} realistic rows from test.csv for benchmarking "
                          f"(cycled through repeatedly to reach {N_TIMED} timed iterations per tier).")

    # --- Tier 1 ---
    tier1_model = xgb.XGBClassifier()
    tier1_model.load_model(TIER1_MODEL_PATH)
    tier1_timings = benchmark_tier1(tier1_model, sample_rows, report_lines)
    tier1_summary = summarize(tier1_timings, "TIER 1 (Fast-Path Binary)", report_lines, target_ms=TIER1_TARGET_MS)

    # --- Tier 3 ---
    tier3_model = xgb.XGBClassifier()
    tier3_model.load_model(TIER3_MODEL_PATH)
    tier3_timings = benchmark_tier3(tier3_model, sample_rows, report_lines)
    tier3_summary = summarize(tier3_timings, "TIER 3 (Full Multi-Class)", report_lines, target_ms=None)
    report_lines.append("\n(No direct target comparison for Tier 3 - its '<5 min' architecture "
                          "target refers to the BATCH WINDOW CADENCE, not per-row inference "
                          "speed, so it is not comparable to this per-row figure.)")

    combined = pd.concat([tier1_timings, tier3_timings], ignore_index=True)
    combined.to_csv(OUTPUT_RAW_CSV, index=False)
    report_lines.append(f"\nRaw per-iteration timings saved to: {OUTPUT_RAW_CSV}")

    report_text = "\n".join(report_lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)
    print(f"\nReport saved to: {OUTPUT_REPORT}")

    # ============================================================
    # Save machine-readable JSON summary
    # ============================================================
    json_summary = {
        "n_warmup": N_WARMUP,
        "n_timed": N_TIMED,
        "tier1": tier1_summary,
        "tier3": tier3_summary,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(json_summary, f, indent=4)
    print(f"[+] Saved machine-readable latency summary to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
