import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- PATH CONFIGURATION ---
BASE_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software"
PROCESSED_DIR = os.path.join(BASE_DIR, "Data Set", "processed")
RESULTS_DIR = os.path.join(PROCESSED_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "Data Set", "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
TABLES_DIR = os.path.join(REPORTS_DIR, "latex_tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

SHAP_CSV = os.path.join(PROCESSED_DIR, "shap_explainability_summary.csv")

TIER1_BASELINE_FILE = os.path.join(RESULTS_DIR, "tier1_baseline_results.json")
TIER1_ADMO_FILE = os.path.join(RESULTS_DIR, "tier1_admo_results.json")
TIER3_BASELINE_FILE = os.path.join(RESULTS_DIR, "tier3_baseline_results.json")
TIER3_ADMO_FILE = os.path.join(RESULTS_DIR, "tier3_admo_results.json")
REALTIME_METRICS_FILE = os.path.join(RESULTS_DIR, "realtime_metrics.json")


def load_json_or_none(path, label):
    if not os.path.exists(path):
        print(f"    [!] {label} missing at {path}. Run the corresponding script first. Skipping dependent output.")
        return None
    with open(path, 'r') as f:
        return json.load(f)


def generate_shap_bar_chart():
    print("[*] Generating SHAP Feature Importance Shift Figure...")
    if not os.path.exists(SHAP_CSV):
        print("    [!] SHAP CSV missing. Skipping plot.")
        return

    df_shap = pd.read_csv(SHAP_CSV).head(8)  # Top 8 features

    plt.figure(figsize=(10, 5), dpi=300)
    bar_width = 0.35
    x = np.arange(len(df_shap))

    plt.bar(x - bar_width/2, df_shap['Baseline_Mean_Abs_SHAP'], width=bar_width, label='Baseline Model (Offline)', color='#d9534f', alpha=0.85)
    plt.bar(x + bar_width/2, df_shap['ADMO_Adapted_Mean_Abs_SHAP'], width=bar_width, label='ADMO Adapted Model (Live)', color='#0275d8', alpha=0.85)

    plt.xlabel('Golden 12 Features', fontsize=11, fontweight='bold')
    plt.ylabel('Mean |SHAP Value| (Impact on Prediction)', fontsize=11, fontweight='bold')
    plt.title('Feature Attribution Shift: Baseline vs. ADMO Domain Adaptation', fontsize=12, fontweight='bold', pad=15)
    plt.xticks(x, df_shap['Feature'], rotation=25, ha='right', fontsize=9)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "figure_shap_feature_shift.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"    [+] Saved Figure: {fig_path}")


def generate_confusion_matrix_comparison():
    print("[*] Generating Baseline vs ADMO Confusion Matrix Figure (Tier 1)...")

    baseline = load_json_or_none(TIER1_BASELINE_FILE, "Tier 1 baseline results (run script 11 first)")
    admo = load_json_or_none(TIER1_ADMO_FILE, "Tier 1 ADMO-adapted results (run script 12 first)")

    if baseline is None or admo is None:
        print("    [!] Skipping confusion matrix comparison figure — missing prerequisite results.")
        return

    cm_baseline = np.array(baseline['confusion_matrix'])
    cm_admo = np.array(admo['confusion_matrix'])

    if cm_baseline.shape != (2, 2) or cm_admo.shape != (2, 2):
        print("    [!] Confusion matrices are not both 2x2 (some class may be entirely absent). "
              "Skipping the side-by-side figure to avoid a misleading plot.")
        return

    base_labels = baseline['confusion_matrix_labels']
    admo_labels = admo['confusion_matrix_labels']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Reds', ax=axes[0],
                xticklabels=[f"Pred {l}" for l in base_labels],
                yticklabels=[f"True {l}" for l in base_labels], cbar=False)
    axes[0].set_title(
        f"Phase 5 Zero-Shot Baseline\n"
        f"(Accuracy: {baseline['accuracy']*100:.2f}% | DoS Recall: {baseline['dos_recall']*100:.2f}%)",
        fontsize=10, fontweight='bold'
    )

    sns.heatmap(cm_admo, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=[f"Pred {l}" for l in admo_labels],
                yticklabels=[f"True {l}" for l in admo_labels], cbar=False)
    axes[1].set_title(
        f"Phase 5 ADMO Adapted Model\n"
        f"(Accuracy: {admo['accuracy']*100:.2f}% | DoS Recall: {admo['dos_recall']*100:.2f}%)",
        fontsize=10, fontweight='bold'
    )

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "figure_confusion_matrix_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"    [+] Saved Figure: {fig_path}")


def generate_latex_tables():
    print("[*] Generating Publication-Ready LaTeX Code Tables...")

    baseline = load_json_or_none(TIER1_BASELINE_FILE, "Tier 1 baseline results (run script 11 first)")
    admo = load_json_or_none(TIER1_ADMO_FILE, "Tier 1 ADMO-adapted results (run script 12 first)")
    realtime = load_json_or_none(REALTIME_METRICS_FILE, "Real-time inference metrics (run script 13 first)")

    # --- Table 1: Model Performance Summary ---
    # NOTE: built with plain token replacement, not Python's % operator, because
    # the LaTeX template itself contains literal "\%" characters (LaTeX's escape
    # for a percent sign). Mixing those with Python's %-formatting causes a
    # ValueError ("unsupported format character") since Python tries to parse
    # every bare "%" as the start of a format spec.
    if baseline is not None and admo is not None:
        latex_perf_template = r"""\begin{table}[htbp]
\centering
\caption{Performance Comparison of Zero-Shot Baseline vs. ADMO Adapted Pipeline on Live Traffic}
\label{tab:admo_performance_comparison}
\begin{tabular}{lcccc}
\hline
\textbf{Pipeline Configuration} & \textbf{Accuracy (\%)} & \textbf{DoS Precision} & \textbf{DoS Recall} & \textbf{DoS F1-Score} \\ \hline
Static Zero-Shot Baseline & __BASE_ACC__\% & __BASE_PREC__ & __BASE_REC__ & __BASE_F1__ \\
\textbf{ADMO Adapted (Ours)} & \textbf{__ADMO_ACC__\%} & \textbf{__ADMO_PREC__} & \textbf{__ADMO_REC__} & \textbf{__ADMO_F1__} \\ \hline
\end{tabular}
\end{table}
"""
        latex_perf = (latex_perf_template
                      .replace("__BASE_ACC__", f"{baseline['accuracy'] * 100:.2f}")
                      .replace("__BASE_PREC__", f"{baseline['dos_precision']:.4f}")
                      .replace("__BASE_REC__", f"{baseline['dos_recall']:.4f}")
                      .replace("__BASE_F1__", f"{baseline['dos_f1']:.4f}")
                      .replace("__ADMO_ACC__", f"{admo['accuracy'] * 100:.2f}")
                      .replace("__ADMO_PREC__", f"{admo['dos_precision']:.4f}")
                      .replace("__ADMO_REC__", f"{admo['dos_recall']:.4f}")
                      .replace("__ADMO_F1__", f"{admo['dos_f1']:.4f}"))

        t1_path = os.path.join(TABLES_DIR, "table_admo_performance.tex")
        with open(t1_path, 'w') as f:
            f.write(latex_perf)
        print(f"    [+] Saved LaTeX Table: {t1_path}")
    else:
        print("    [!] Skipping performance table — missing prerequisite results.")

    # --- Table 2: Real-Time Performance ---
    if realtime is not None:
        latex_realtime_template = r"""\begin{table}[htbp]
\centering
\caption{Real-Time Streaming Inference \& Throughput Metrics}
\label{tab:realtime_metrics}
\begin{tabular}{lc}
\hline
\textbf{Metric} & \textbf{Measured Value} \\ \hline
Total Telemetry Flows Evaluated & __TOTAL_FLOWS__ \\
Tier 1 Fast-Path Offload Ratio & __T1_PCT__\% (__T1_HITS__ / __TOTAL_FLOWS__) \\
Tier 3 Slow-Path Escalation Ratio & __T3_PCT__\% (__T3_EVALS__ / __TOTAL_FLOWS__) \\
Average Micro-Batch Per-Flow Latency & __AVG_LAT__ $\mu$s \\
Streaming Throughput Capacity & __THROUGHPUT__ flows/sec \\ \hline
\end{tabular}
\end{table}
"""
        latex_realtime = (latex_realtime_template
                           .replace("__TOTAL_FLOWS__", str(realtime['total_flows']))
                           .replace("__T1_PCT__", f"{realtime['tier1_fastpath_pct']:.2f}")
                           .replace("__T1_HITS__", str(realtime['tier1_fastpath_hits']))
                           .replace("__T3_PCT__", f"{realtime['tier3_evaluations_pct']:.2f}")
                           .replace("__T3_EVALS__", str(realtime['tier3_evaluations']))
                           .replace("__AVG_LAT__", f"{realtime['avg_latency_us']:.2f}")
                           .replace("__THROUGHPUT__", f"{realtime['throughput_fps']:.2f}"))

        t2_path = os.path.join(TABLES_DIR, "table_realtime_metrics.tex")
        with open(t2_path, 'w') as f:
            f.write(latex_realtime)
        print(f"    [+] Saved LaTeX Table: {t2_path}")
    else:
        print("    [!] Skipping real-time metrics table — missing prerequisite results.")


def main():
    print("=" * 60)
    print("--- GENERATING THESIS PUBLICATION ARTIFACTS ---")
    print("=" * 60)
    generate_shap_bar_chart()
    generate_confusion_matrix_comparison()
    generate_latex_tables()
    print("\n[+] Phase 6 thesis artifacts generation complete (see [!] warnings above for any skipped items).")


if __name__ == "__main__":
    main()
