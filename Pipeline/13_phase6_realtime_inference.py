import os
import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime

# --- PATH CONFIGURATION ---
BASE_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software"
PROCESSED_DIR = os.path.join(BASE_DIR, "Data Set", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "Data Set", "models")
RESULTS_DIR = os.path.join(PROCESSED_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LIVE_FLOWS_FILE = os.path.join(PROCESSED_DIR, "phase5_flows.csv")
TIER1_MODEL_FILE = os.path.join(MODELS_DIR, "tier1_admo_adapted.json")
TIER3_MODEL_FILE = os.path.join(MODELS_DIR, "tier3_admo_adapted.json")
ALERTS_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "realtime_alerts.json")
REALTIME_METRICS_FILE = os.path.join(RESULTS_DIR, "realtime_metrics.json")

# --- GOLDEN 12 FEATURE MAPPING ---
COLUMN_MAPPING = {
    'fwd_header_len': 'Fwd Header Length',
    'bwd_header_len': 'Bwd Header Length',
    'fwd_psh_flags': 'Fwd PSH Flags',
    'ack_flag_cnt': 'ACK Flag Count',
    'init_fwd_win_byts': 'Init_Win_bytes_forward',
    'flow_duration': 'Flow Duration',
    'flow_pkts_s': 'Flow Packets/s',
    'flow_byts_s': 'Flow Bytes/s',
    'fwd_pkt_len_max': 'Fwd Packet Length Max',
    'flow_iat_mean': 'Flow IAT Mean',
    'fwd_iat_max': 'Fwd IAT Max',
    'bwd_iat_min': 'Bwd IAT Min'
}

GOLDEN_12 = [
    'Fwd Header Length', 'Bwd Header Length', 'Fwd PSH Flags', 'ACK Flag Count',
    'Init_Win_bytes_forward', 'Flow Duration', 'Flow Packets/s', 'Flow Bytes/s',
    'Fwd Packet Length Max', 'Flow IAT Mean', 'Fwd IAT Max', 'Bwd IAT Min'
]

INV_CLASS_MAPPING = {
    0: 'BENIGN',
    1: 'DoS_DDoS',
    2: 'BRUTE_FORCE',
    3: 'INFILTRATION',
    4: 'BOTNET',
    5: 'WEB_ATTACK'
}


class RealTimeNIDSInferenceEngine:
    def __init__(self):
        print("[*] Initializing Real-Time NIDS Hierarchical Inference Engine...")

        self.tier1_model = xgb.Booster()
        self.tier1_model.load_model(TIER1_MODEL_FILE)
        print(f"    [+] Loaded Tier 1 Fast-Path Binary Model: {os.path.basename(TIER1_MODEL_FILE)}")

        self.tier3_model = xgb.Booster()
        self.tier3_model.load_model(TIER3_MODEL_FILE)
        print(f"    [+] Loaded Tier 3 Full Multi-Class Model: {os.path.basename(TIER3_MODEL_FILE)}")

    def preprocess_flow(self, flow_series):
        flow_dict = flow_series.to_dict()

        renamed_flow = {}
        for raw_k, val in flow_dict.items():
            mapped_k = COLUMN_MAPPING.get(raw_k, raw_k)
            renamed_flow[mapped_k] = val

        for col in GOLDEN_12:
            v = renamed_flow.get(col, 0.0)
            if pd.isna(v) or v == np.inf or v == -np.inf:
                renamed_flow[col] = 0.0
            else:
                renamed_flow[col] = float(v)

        time_cols = ['Flow Duration', 'Flow IAT Mean', 'Fwd IAT Max', 'Bwd IAT Min']
        for c in time_cols:
            renamed_flow[c] = renamed_flow[c] * 1_000_000.0

        features_vector = [renamed_flow[c] for c in GOLDEN_12]
        return features_vector, renamed_flow

    def process_live_stream(self, batch_size=100):
        print(f"\n[*] Starting Stream Processing on {LIVE_FLOWS_FILE}...")
        df_flows = pd.read_csv(LIVE_FLOWS_FILE)
        total_flows = len(df_flows)
        print(f"[*] Ingested {total_flows} raw telemetry flows for stream simulation.\n")

        alerts_generated = []
        tier1_fastpath_hits = 0
        tier3_evaluations = 0

        start_total_time = time.time()

        for idx, row in df_flows.iterrows():
            flow_start_time = time.time()
            features, flow_data = self.preprocess_flow(row)

            dmatrix = xgb.DMatrix(np.array([features]), feature_names=GOLDEN_12)

            t1_prob = float(self.tier1_model.predict(dmatrix)[0])
            latency_us = (time.time() - flow_start_time) * 1_000_000.0

            if t1_prob > 0.5:
                tier1_fastpath_hits += 1
                alert = {
                    'timestamp': str(flow_data.get('timestamp', datetime.now().isoformat())),
                    'flow_id': idx,
                    'src_ip': flow_data.get('src_ip', 'N/A'),
                    'dst_ip': flow_data.get('dst_ip', 'N/A'),
                    'dst_port': int(flow_data.get('dst_port', 0)),
                    'severity': 'CRITICAL',
                    'predicted_class': 'DoS_DDoS',
                    'confidence': round(t1_prob, 4),
                    'pipeline_tier': 'Tier 1 (Fast-Path)',
                    'latency_us': round(latency_us, 2)
                }
                alerts_generated.append(alert)
            else:
                tier3_evaluations += 1
                t3_probs = self.tier3_model.predict(dmatrix)[0]
                pred_class_idx = int(np.argmax(t3_probs))
                pred_class_name = INV_CLASS_MAPPING.get(pred_class_idx, 'UNKNOWN')
                confidence = float(t3_probs[pred_class_idx])

                latency_us = (time.time() - flow_start_time) * 1_000_000.0

                if pred_class_name != 'BENIGN':
                    alert = {
                        'timestamp': str(flow_data.get('timestamp', datetime.now().isoformat())),
                        'flow_id': idx,
                        'src_ip': flow_data.get('src_ip', 'N/A'),
                        'dst_ip': flow_data.get('dst_ip', 'N/A'),
                        'dst_port': int(flow_data.get('dst_port', 0)),
                        'severity': 'HIGH' if pred_class_name in ['BRUTE_FORCE', 'BOTNET'] else 'MEDIUM',
                        'predicted_class': pred_class_name,
                        'confidence': round(confidence, 4),
                        'pipeline_tier': 'Tier 3 (Slow-Path)',
                        'latency_us': round(latency_us, 2)
                    }
                    alerts_generated.append(alert)

        total_elapsed_sec = time.time() - start_total_time
        avg_latency_us = (total_elapsed_sec / total_flows) * 1_000_000.0
        throughput_fps = total_flows / total_elapsed_sec
        tier1_pct = tier1_fastpath_hits / total_flows * 100
        tier3_pct = tier3_evaluations / total_flows * 100

        print("=" * 60)
        print("--- REAL-TIME INFERENCE PERFORMANCE SUMMARY ---")
        print("=" * 60)
        print(f"Total Flows Processed         : {total_flows}")
        print(f"Tier 1 Fast-Path DoS Triggers : {tier1_fastpath_hits} ({tier1_pct:.2f}%)")
        print(f"Tier 3 Slow-Path Evaluated   : {tier3_evaluations} ({tier3_pct:.2f}%)")
        print(f"Total Security Alerts Emitted : {len(alerts_generated)}")
        print(f"Overall Processing Time      : {total_elapsed_sec:.4f} seconds")
        print(f"Average Per-Flow Latency     : {avg_latency_us:.2f} microseconds (µs)")
        print(f"System Stream Throughput      : {throughput_fps:.2f} flows/sec")

        with open(ALERTS_OUTPUT_FILE, 'w') as f:
            json.dump(alerts_generated, f, indent=4)
        print(f"\n[+] Real-time alerts successfully logged to: {ALERTS_OUTPUT_FILE}")

        # --- SAVE SUMMARY METRICS FOR THESIS ARTIFACTS ---
        realtime_metrics = {
            'total_flows': int(total_flows),
            'tier1_fastpath_hits': int(tier1_fastpath_hits),
            'tier1_fastpath_pct': tier1_pct,
            'tier3_evaluations': int(tier3_evaluations),
            'tier3_evaluations_pct': tier3_pct,
            'total_alerts_emitted': int(len(alerts_generated)),
            'total_elapsed_sec': total_elapsed_sec,
            'avg_latency_us': avg_latency_us,
            'throughput_fps': throughput_fps
        }
        with open(REALTIME_METRICS_FILE, 'w') as f:
            json.dump(realtime_metrics, f, indent=4)
        print(f"[+] Saved real-time summary metrics to: {REALTIME_METRICS_FILE}")

        if len(alerts_generated) > 0:
            print("\n=== SAMPLE EMITTED REAL-TIME ALERT ===")
            print(json.dumps(alerts_generated[0], indent=4))


def main():
    engine = RealTimeNIDSInferenceEngine()
    engine.process_live_stream()


if __name__ == "__main__":
    main()
