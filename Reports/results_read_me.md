# ADMO Framework: Comprehensive Experimental Results & Benchmarks

**Project Title:** Latency-Aware Network Protection: Orchestrating a Universal IDS with a Machine Learning Middleware in ISP Environments  
**Candidate:** Ali Ahson (Matr. B33000053)  
**Supervisor:** Prof. Fiammetta Marulli  
**Degree Program:** Master’s Degree in Data Science (Specialization: Machine Learning & Artificial Intelligence)  
**Institution:** Università degli Studi della Campania “Luigi Vanvitelli” · Dipartimento di Matematica e Fisica  
**Academic Year:** 2025/2026  

---

##  EXECUTIVE SUMMARY

This document summarizes the quantitative results and experimental findings obtained across all 6 phases of the **ADMO (Adaptive Domain-Adapted Middleware Orchestrator)** research software pipeline. 

ADMO resolves the **Latency-Inspection Paradox** in high-speed Internet Service Provider (ISP) networks by placing machine learning evaluation out-of-band on SPAN-mirrored telemetry. By decoupling the Data Plane from the Analytics Plane, ADMO adds **0.00 ms of synchronous forwarding delay** to live traffic while maintaining sub-millisecond threat detection and automated kernel-level firewall mitigation (`pfctl`).

---

## 1. KEY PERFORMANCE COMPARISON: BASELINE VS. ADMO ADAPTED MODEL

### The Live Domain Shift Discovery (Phase 5)
When a static zero-shot baseline model trained on historical benchmark datasets (CICIDS-2017 / CSE-CIC-IDS-2018) was deployed on live pfSense network traffic (`hping3` Linux SYN floods), it suffered a **100% detection failure**.

* **Root Cause Analysis:** Offline decision trees overfitted to synthetic Windows LOIC tool fingerprints (`Init_Win_bytes_forward` = 14,788 Bytes, long flow durations) rather than invariant rate dynamics. When live Linux `hping3` packets arrived with `Init_Win_bytes_forward` = 512 Bytes in short micro-flows, the static model classified 100% of attack traffic as `BENIGN`.
* **ADMO Domain Adaptation Fix:** Hybrid retraining mixing a 30% sample of live lab telemetry (~1,300 flows) into the 2.6-million benchmark dataset successfully bridged the zero-shot domain gap.

### Quantitative Results on Live Held-Out Test Set (3,039 Live Flows)

| Evaluation Metric | Static Zero-Shot Baseline | ADMO Adapted Model (Ours) | Delta Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Live Accuracy** | **4.79%** | **95.16%** | **+90.37%** 📈 |
| **DoS / DDoS Precision** | **0.0000** | **0.9699** | **+0.9699** 📈 |
| **DoS / DDoS Recall** | **0.0000** | **0.9796** | **+0.9796** 📈 |
| **DoS / DDoS F1-Score** | **0.0000** | **0.9747** | **+0.9747** 📈 |

#### Confusion Matrix Breakdown (ADMO Adapted Model)
* **True DoS Flows Correctly Identified:** 2,834 out of 2,893 flows (**97.96% Recall**)
* **False Negatives (Missed DoS):** 59 out of 2,893 flows
* **True BENIGN Flows Correctly Handled:** 58 out of 146 flows

---

## 2. REAL-TIME STREAMING & SYSTEM VIABILITY METRICS (PHASE 6)

Evaluating real-time streaming performance across 4,341 live telemetry flows confirmed that the hierarchical cascade engine operates well within ISP SLA latency constraints.

| Performance Metric | Measured Value | Operational Significance |
| :--- | :---: | :--- |
| **Total Live Flows Processed** | **4,341 flows** | Full live micro-batch stream evaluation |
| **Tier 1 Fast-Path Offload Ratio** | **96.45%** (4,187 / 4,341) | High-volume DoS filtered at sub-ms binary tier |
| **Tier 3 Escalation Ratio** | **3.55%** (154 / 4,341) | Complex ambiguous flows escalated to multi-class |
| **Average Per-Flow Inference Latency** | **781.05 µs** | Well under the 1.0 ms real-time ISP budget |
| **System Streaming Throughput** | **1,280.33 flows/sec** | Sustained pure Python throughput on single CPU core |
| **Added Forwarding Latency** | **0.00 ms** | Guaranteed by out-of-band SPAN mirror design |
| **System Resource Footprint** | **~1.8 GB RAM** | 40–65% CPU load on 1 core across all services |

---

## 3. EXPLAINABLE AI (XAI) & TREESHAP FEATURE ATTRIBUTION SHIFT

To mathematically explain why the baseline failed and why ADMO succeeded, native XGBoost C++ TreeSHAP (`pred_contribs=True`) was executed across all live flows.

### Additive Feature Attribution Formulation
Based on cooperative game theory, Shapley values decompose model output into additive marginal contributions:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
where $\phi_0$ is the base expected model prediction and $\phi_i(x)$ is the exact marginal contribution (impact score) of feature $i$.

### SHAP Feature Importance Comparison (Mean |SHAP Value|)

| Feature Name | Baseline Mean \|SHAP\| | ADMO Adapted Mean \|SHAP\| | SHAP Shift Delta | Attribution Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| `Init_Win_bytes_forward` | **5.417895** | **0.563375** | **-4.854520** 📉 | **Huge Drop:** Abandoned fragile socket fingerprint |
| `Flow Duration` | **0.748713** | **1.105181** | **+0.356468** 📈 | **Primary Driver:** Pivoted to invariant flow time |
| `ACK Flag Count` | **0.385409** | **0.502879** | **+0.117470** 📈 | **Increased Reliance:** Rate-invariant TCP control count |
| `Fwd Packet Length Max` | **3.365781** | **0.866088** | **-2.499693** 📉 | **Reduced Reliance:** Reduced tool packet size bias |
| `Flow Bytes/s` | **1.628483** | **0.644373** | **-0.984110** 📉 | Normalized rate metric adjustment |
| `Flow IAT Mean` | **0.192620** | **0.222274** | **+0.029654** 📈 | Stable inter-arrival timing weight |

**Mathematical Proof:** TreeSHAP proves that hybrid retraining forced XGBoost to drop its single largest decision driver (`Init_Win_bytes_forward` dropped by **-4.85**) and pivot toward true behavioral flow dynamics (`Flow Duration` SHAP rose to **1.11**).

---

## 4. CLASS IMBALANCE & GENERATIVE AUGMENTATION (CTGAN + SMOTEENN)

To resolve extreme class imbalance in historical datasets (13.2M Benign vs 2,000 Infiltration samples — 6,000:1 ratio), ADMO coupled **Conditional Tabular GANs (CTGAN)** with GMM pre-training and **SMOTEENN** boundary cleaning.

| Threat Class | Real Flow Records | CTGAN Synthetic Added | Total Augmented Set | Expansion Factor |
| :--- | :---: | :---: | :---: | :---: |
| **BENIGN** | 13,200,000 | — | 13,200,000 | Baseline Majority |
| **DoS / DDoS** | 2,000,000 | — | 2,000,000 | High Volume |
| **BRUTE_FORCE** | 340,000 | — | 340,000 | Moderate |
| **WEB_ATTACK** | 200,000 | +50,000 | 250,000 | 1.25x |
| **BOTNET** | 120,000 | +80,000 | 200,000 | 1.67x |
| **INFILTRATION** | **2,000** | **+78,000** | **80,000** | **40.0x Expansion** 🚀 |

---

## 5. UNIVERSAL ORCHESTRATOR EXECUTION SUMMARY (SURICATA 7 EVE JSON)

The Universal Orchestrator Adapter (`16_suricata_eve_orchestrator.py`) tails `/var/log/suricata/eve.json` streams in real time, extracts Golden 12 vectors, evaluates multi-tier models, and dispatches kernel blocks.

### Sample Live Stream Execution Log
```text
[*] Initializing Suricata EVE JSON Universal Orchestrator...
    [+] Loaded Tier 1 Fast-Path Binary Model: tier1_admo_adapted.json
    [+] Loaded Tier 3 Full Multi-Class Model: tier3_admo_adapted.json
[*] Starting Suricata EVE JSON Stream Reader on: ...\processed\eve.json

-> Parsed Event #1 [10.10.20.5:45210 -> 10.10.10.5:8080]   | Tier 1 DoS Prob: 0.0368 (Passed)
-> Parsed Event #2 [192.168.1.50:51200 -> 10.10.10.5:443]  | Tier 1 DoS Prob: 0.0000 (Passed)
-> Parsed Event #3 [10.10.20.99:44444 -> 10.10.10.5:80]    | Tier 1 DoS Prob: 0.0082 (Passed)
-> Parsed Event #4 [10.10.20.99:44444 -> 10.10.10.5:8080]  | Tier 1 DoS Prob: 0.1322 (Escalated to Tier 3)
   [HIGH THREAT TRIGGERED]
    • Threat Class: DoS_DDoS (Confidence: 0.5942)
    • Pipeline    : Tier 3 (Slow-Path Multi-Class)
    [ACTION EXECUTION]: pfctl -t snort2c -T add 10.10.20.99
-----------------------------------------------------------------
-> Parsed Event #5 [10.10.20.5:44444 -> 10.10.10.5:8080]   | Tier 1 DoS Prob: 0.1893 (Passed)

============================================================
--- SURICATA EVE ORCHESTRATION SUMMARY ---
============================================================
Total Suricata Events Parsed : 5
Automated pfctl IP Blocks   : 1