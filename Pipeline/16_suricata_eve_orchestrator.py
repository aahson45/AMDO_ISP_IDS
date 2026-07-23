import os
import sys
import time
import json
import subprocess
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime

# --- PATH CONFIGURATION ---
BASE_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software"
PROCESSED_DIR = os.path.join(BASE_DIR, "Data Set", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "Data Set", "models")

TIER1_MODEL_FILE = os.path.join(MODELS_DIR, "tier1_admo_adapted.json")
TIER3_MODEL_FILE = os.path.join(MODELS_DIR, "tier3_admo_adapted.json")
EVE_LOG_FILE = os.path.join(PROCESSED_DIR, "eve.json")

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


class SuricataEveOrchestrator:
    def __init__(self, eve_path, dry_run=True):
        self.eve_path = eve_path
        self.dry_run = dry_run
        
        print("[*] Initializing Suricata EVE JSON Universal Orchestrator...")
        
        # Load Tier 1 Model
        self.tier1_model = xgb.Booster()
        self.tier1_model.load_model(TIER1_MODEL_FILE)
        print(f"    [+] Loaded Tier 1 Fast-Path Binary Model: {os.path.basename(TIER1_MODEL_FILE)}")

        # Load Tier 3 Model
        self.tier3_model = xgb.Booster()
        self.tier3_model.load_model(TIER3_MODEL_FILE)
        print(f"    [+] Loaded Tier 3 Full Multi-Class Model: {os.path.basename(TIER3_MODEL_FILE)}")

    def parse_suricata_flow_event(self, eve_record):
        """Parses Suricata EVE JSON 'flow' or 'netflow' event into Golden 12 format."""
        flow = eve_record.get('flow', {})
        tcp_info = eve_record.get('tcp', {})
        
        # Extract Flow Duration (convert seconds/ms to microseconds)
        age_sec = flow.get('age', 0.037)
        duration_us = age_sec * 1_000_000.0 if age_sec > 0 else 1000.0

        pkts_tosc = flow.get('pkts_toserver', 1)
        pkts_tocl = flow.get('pkts_toclient', 0)
        total_pkts = pkts_tosc + pkts_tocl

        bytes_tosc = flow.get('bytes_toserver', 64)
        bytes_tocl = flow.get('bytes_toclient', 0)
        total_bytes = bytes_tosc + bytes_tocl

        # Calculate Rates
        duration_sec = duration_us / 1_000_000.0
        flow_pkts_s = (total_pkts / duration_sec) if duration_sec > 0 else 0.0
        flow_byts_s = (total_bytes / duration_sec) if duration_sec > 0 else 0.0

        # Map to Golden 12 Vector
        golden_vector = {
            'Fwd Header Length': float(pkts_tosc * 40), # IP (20B) + TCP (20B) = 40B per pkt
            'Bwd Header Length': float(pkts_tocl * 40),
            'Fwd PSH Flags': 1.0 if tcp_info.get('psh', False) else 0.0,
            'ACK Flag Count': float(pkts_tosc if tcp_info.get('ack', True) else 0),
            'Init_Win_bytes_forward': float(tcp_info.get('window', 512)),
            'Flow Duration': float(duration_us),
            'Flow Packets/s': float(flow_pkts_s),
            'Flow Bytes/s': float(flow_byts_s),
            'Fwd Packet Length Max': float(bytes_tosc / max(1, pkts_tosc)),
            'Flow IAT Mean': float(duration_us / max(1, total_pkts)),
            'Fwd IAT Max': float(duration_us / max(1, pkts_tosc)),
            'Bwd IAT Min': float(duration_us / max(1, pkts_tocl)) if pkts_tocl > 0 else 0.0
        }

        vector_array = [golden_vector[col] for col in GOLDEN_12]
        meta_5tuple = {
            'src_ip': eve_record.get('src_ip', '0.0.0.0'),
            'src_port': eve_record.get('src_port', 0),
            'dst_ip': eve_record.get('dest_ip', '0.0.0.0'),
            'dst_port': eve_record.get('dest_port', 0),
            'proto': eve_record.get('proto', 'TCP')
        }
        return vector_array, meta_5tuple

    def execute_pfctl_block(self, target_ip):
        """Executes pfctl command to add target IP to pfSense snort2c block table."""
        cmd = f"pfctl -t snort2c -T add {target_ip}"
        if self.dry_run:
            print(f"    [ACTION EXECUTION - DRY RUN]: {cmd}")
        else:
            try:
                res = subprocess.run(["pfctl", "-t", "snort2c", "-T", "add", target_ip], capture_output=True, text=True)
                print(f"    [ACTION EXECUTED - KERNEL BLOCK]: Added {target_ip} to snort2c. Output: {res.stdout.strip()}")
            except Exception as e:
                print(f"    [!] Error executing pfctl: {e}")

    def process_eve_stream(self):
        print(f"[*] Starting Suricata EVE JSON Stream Reader on: {self.eve_path}\n")

        processed_count = 0
        alerts_emitted = 0

        with open(self.eve_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if record.get('event_type') in ['flow', 'netflow', 'alert']:
                    processed_count += 1
                    features, meta = self.parse_suricata_flow_event(record)
                    
                    dmatrix = xgb.DMatrix(np.array([features]), feature_names=GOLDEN_12)

                    # --- TIER 1 EVALUATION ---
                    t1_prob = float(self.tier1_model.predict(dmatrix)[0])

                    # Print Verbose Diagnostics for Every Event
                    print(f"-> Parsed Event #{processed_count} [{meta['src_ip']}:{meta['src_port']} -> {meta['dst_ip']}:{meta['dst_port']}] | Tier 1 DoS Prob: {t1_prob:.4f}")

                    if t1_prob > 0.5:
                        alerts_emitted += 1
                        print(f"   [CRITICAL THREAT TRIGGERED]")
                        print(f"    • Threat Class: DoS_DDoS (Confidence: {t1_prob:.4f})")
                        print(f"    • Pipeline    : Tier 1 (Fast-Path Binary)")
                        self.execute_pfctl_block(meta['src_ip'])
                        print("-" * 65)
                    else:
                        # --- TIER 3 EVALUATION ---
                        t3_probs = self.tier3_model.predict(dmatrix)[0]
                        pred_idx = int(np.argmax(t3_probs))
                        pred_class = INV_CLASS_MAPPING.get(pred_idx, 'BENIGN')
                        confidence = float(t3_probs[pred_idx])

                        if pred_class != 'BENIGN':
                            alerts_emitted += 1
                            print(f"   [HIGH THREAT TRIGGERED]")
                            print(f"    • Threat Class: {pred_class} (Confidence: {confidence:.4f})")
                            print(f"    • Pipeline    : Tier 3 (Slow-Path Multi-Class)")
                            self.execute_pfctl_block(meta['src_ip'])
                            print("-" * 65)

        print("\n" + "=" * 60)
        print("--- SURICATA EVE ORCHESTRATION SUMMARY ---")
        print("=" * 60)
        print(f"Total Suricata Events Parsed : {processed_count}")
        print(f"Automated pfctl IP Blocks   : {alerts_emitted}")


if __name__ == "__main__":
    orchestrator = SuricataEveOrchestrator(eve_path=EVE_LOG_FILE, dry_run=True)
    orchestrator.process_eve_stream()