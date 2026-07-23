# Latency-Aware Network Protection: A Hardware-Accelerated ML Middle Layer for Universal IDS Orchestration in ISP Environments

**M.Sc. Thesis in Data Science**  
**University of Campania "Luigi Vanvitelli", Italy**  
**Industry Partner:** TIM S.p.A. (Telecom Italia)  
**Author:** Ali Ahson ([aahson45@gmail.com](mailto:aahson45@gmail.com))  
**Supervisor:** Prof. Fiammetta Marulli  
**Academic Year:** 2025/2026  

---

## 📌 1. Executive Summary

Internet Service Providers (ISPs) face an operational trade-off: either perform shallow packet forwarding to maintain ultra-low network latency, or execute deep packet inspection (DPI) to detect complex cyber threats. Running inline intrusion detection appliances directly in the forwarding path introduces unacceptable latency and jitter (5.0–50.0 ms), forcing operators into passive monitoring rather than active threat blocking.

**ADMO (Adaptive Domain-Adapted Middleware Orchestrator)** is an out-of-band machine learning middleware designed to eliminate this trade-off[cite: 1, 4, 5]. Operating on SPAN-mirrored packet telemetry off the critical forwarding path, ADMO adds **0.00 ms of synchronous forwarding delay** while achieving **95.16% accuracy** and **97.96% DoS recall** on live network traffic.

---

## 📐 2. System Architecture

```text
                             [ INTERNET / WAN ]
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │  pfSense Firewall (Netgate VM)  │ <--- Data Plane (<0.10 ms)
                    └─────────────────────────────────┘
                             │               │
      (Mirrored SPAN Port)   │               │ (Forwarded Traffic)
                             ▼               ▼
┌─────────────────────────────────────────┐  [ LAN / VICTIM HOST ]
│  Suricata 7 / CICFlowMeter Receiver     │
└─────────────────────────────────────────┘
                             │
                             ▼ (EVE JSON / Flow Streams)[cite: 1]
┌──────────────────────────────────────────────────────────────────────────┐
│                    ANALYTICS PLANE (ADMO MIDDLEWARE)                     │
│                                                                          │
│   ┌────────────────────────┐         ┌───────────────────────────────┐   │
│   │  Tier 1 Fast-Path      │ ------> │  Tier 3 Multi-Class           │   │
│   │  Binary XGBoost        │ (Pass)  │  XGBoost / LightGBM Classifier│   │
│   └────────────────────────┘         └───────────────────────────────┘   │
│               │ (Threat Flagged)                     │ (Threat Flagged)  │
│               └──────────────────┬───────────────────┘                   │
│                                  ▼                                       │
│                ┌───────────────────────────────────┐                     │
│                │ Universal Orchestrator Adapter    │                     │
│                └───────────────────────────────────┘                     │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼ (Asynchronous Kernel Call)[cite: 1]
                  [ pfctl -t snort2c -T add <Attacker_IP> ][cite: 1]

```
---
🗂️ 3. Repository Directory Structure        
---
``` text 
├── README.md                              # Main Research & Execution Guide
├── pipeline/                              # Processing & Execution Scripts
│   ├── 00_diagnose.py                     # Phase 1: Data Integrity & Health Diagnostics
│   ├── 01_label_normalization.py          # Phase 1: Taxonomy Normalization Across Datasets
│   ├── 02_merge_and_clean.py              # Phase 1: Cleaning & Infinite Value Filtering
│   ├── 03_feature_selection.py            # Phase 1: Random Forest & ANOVA Golden 12 Selection
│   ├── 04_class_balance_report.py         # Phase 1: Multi-Class Imbalance Analysis
│   ├── 05_ctgan_smoteenn_augment.py       # Phase 1: Generative CTGAN + SMOTEENN Minority Synthesis
│   ├── 06_train_test_split.py             # Phase 1: Stratified Temporal Train/Val/Test Split
│   ├── 07_train_fastpath_binary.py        # Phase 2: Tier 1 Fast-Path Binary XGBoost Training
│   ├── 08_train_batch_aggregator.py       # Phase 2: Tier 2 5-Minute Window Flow Aggregation
│   ├── 09_train_full_classifier.py        # Phase 2: Tier 3 Multi-Class XGBoost & LightGBM Training
│   ├── 10_latency_bench.py                # Phase 2: Type A ML Pipeline Benchmark (MP1–MP4)
│   ├── 11_phase5_evaluate_lab_flows.py    # Phase 5: Zero-Shot Static Model Failure Diagnostic
│   ├── 12_phase5_admo_domain_adaptation.py# Phase 5: ADMO Hybrid Retraining & Adaptation
│   ├── 13_phase6_realtime_inference.py    # Phase 6: Micro-Batch Real-Time Streaming Engine
│   ├── 14_phase6_shap_explainability.py   # Phase 6: Native C++ TreeSHAP Attribution
│   ├── 15_phase6_generate_thesis_artifacts.py # Phase 6: LaTeX Tables & PNG Figure Generator
│   ├── 16_suricata_eve_orchestrator.py    # Component 4: Suricata 7 EVE JSON Adapter
│   ├── bench_baseline_latency.py          # Phase 3: Type B Network Forwarding Latency
│   └── generate_lab_traffic.py            # Phase 4: Kali Linux Traffic Generator Node
│
├── models/                                # Serialized Model Binaries
│   ├── tier1_fastpath_binary.json         # Static Baseline Tier 1 Binary Model
│   ├── tier1_admo_adapted.json            # Live Adapted Tier 1 Model
│   ├── tier3_full_classifier_xgboost.json # Static Baseline Tier 3 Model
│   ├── tier3_full_classifier_lightgbm.txt # Static LightGBM Multi-Class Model
│   └── tier3_admo_adapted.json            # Live Adapted Tier 3 Model
│
├── data/                                  # PCAP Captures & JSON Manifests
│   ├── wan_capture.pcap                   # WAN Interface Capture (em0)
│   ├── lan_capture.pcap                   # LAN Interface Capture (em1)
│   ├── phase5_mixed_traffic.pcap          # Live Lab Traffic Run
│   ├── 06_split_manifest.json             # Train/Val/Test Split Metadata
│   └── label_mapping.json                 # Class Taxonomy Mappings
│
├── reports/                               # Execution Summary Reports
│   ├── 00_diagnose_report.txt
│   ├── 01_label_report.txt
│   ├── 02_merge_report.txt
│   ├── 03_feature_selection_report.txt
│   ├── 04_class_balance_report.txt
│   ├── 05_augmentation_report.txt
│   ├── 06_train_test_split_report.txt
│   ├── 07_tier1_fastpath_report.txt
│   ├── 08_batch_aggregator_report.txt
│   ├── 09_tier3_full_classifier_report.txt
│   ├── 10_latency_bench_report.txt
│   └── phase3_baseline_latency_report.txt
│
└── artifacts/                             # Thesis Figures & LaTeX Source Tables
    ├── figure_confusion_matrix_comparison.png
    ├── figure_shap_feature_shift.png
    ├── table_admo_performance.tex
    └── table_realtime_metrics.tex
```
---
💻 4. Prerequisites & Installation
---
Environment Requirements
Analytics Host: Windows 10/11 or FreeBSD running Python 3.10+

Firewall Appliance: pfSense 2.7.x (FreeBSD 14)

Traffic Generator: Kali Linux 2024.x (hping3, hydra, python3-scapy, python3-requests)[cite: 7]

Dependency Installation
Install the python libraries required across the pipeline
```
& "C:\Users\usid\AppData\Local\Programs\Python\Python310\python.exe" -m pip install numpy pandas xgboost lightgbm scikit-learn matplotlib seaborn scapy python-pptx
```
---
## 📊 5. Summary of Key Experimental Results
---
Live Classification Performance (3,039 Held-Out Live Test Flows)

|Evaluation MetricStatic | Zero-Shot Baseline | ADMO Adapted Model | Performance Improvement|
| --- | --- | --- | --- |
|Overall Live Accuracy | 4.79%  | 95.16% | +90.37% 📈|
|DoS / DDoS  Precision | 0.0000 | 0.9699| +0.9699 📈|
|DoS / DDoS  Recall | 0.0000| 0.9796 | +0.9796 📈|
|DoS / DDoS F1-Score| 0.0000| 0.9747 | +0.9747 📈|

True DoS Flows Identified: 2,834 / 2,893 flows (97.96% Recall)

True BENIGN Flows Handled: 58 / 146 flows
---
Real-Time Streaming & SLA Metrics
---
| **Metric Name**                    | **Measured Value**     | **Operational Context**                             |
|------------------------------------|------------------------|-----------------------------------------------------|
| Total Live Telemetry Flows         | 4,341 Flows            | Evaluated under real-time micro-batch streaming     |
| Tier 1 Fast-Path Offload Ratio     | 96.45% (4,187 / 4,341) | DoS filtered at sub-millisecond binary tier         |
| Tier 3 Slow-Path Escalation Ratio  | 3.55% (154 / 4,341)    | Ambiguous flows escalated to multi-class tier       |
| Average Per-Flow Inference Latency | 781.05 µs              | Well within the 1.0 ms real-time SLA budget         |
| System Streaming Capacity          | 1,280.33 flows/sec     | Sustained pure Python throughput on single CPU core |
| Added Forwarding Latency           | 1,280.33 flows/sec     | Guaranteed by out-of-band SPAN mirror design        |


---
##⚡ 7. All-In-One Automated Execution Script
---
Run this PowerShell command to execute all pipeline stages sequentially:
``` code
$PYTHON="C:\Users\usid3\AppData\Local\Programs\Python\Python310\python.exe"
$BASE="C:\Users\usid3\Downloads\Ali Ahson Thesis Software\pipeline"

Write-Host "=== STARTING FULL ADMO PIPELINE EXECUTION ===" -ForegroundColor Green

& $PYTHON "$BASE\09_train_full_classifier.py"[cite: 10]
& $PYTHON "$BASE\10_latency_bench.py"[cite: 9]
& $PYTHON "$BASE\bench_baseline_latency.py"[cite: 8]
& $PYTHON "$BASE\11_phase5_evaluate_lab_flows.py"[cite: 6]
& $PYTHON "$BASE\12_phase5_admo_domain_adaptation.py"[cite: 5]
& $PYTHON "$BASE\13_phase6_realtime_inference.py"[cite: 4]
& $PYTHON "$BASE\14_phase6_shap_explainability.py"[cite: 3]
& $PYTHON "$BASE\15_phase6_generate_thesis_artifacts.py"[cite: 2]
& $PYTHON "$BASE\16_suricata_eve_orchestrator.py"[cite: 1]
& $PYTHON "$BASE\generate_deck.py"

Write-Host "=== ALL PIPELINE STAGES COMPLETED SUCCESSFULLY ===" -ForegroundColor Green
```
