import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

# --- PATH CONFIGURATION ---
BASE_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software"
PROCESSED_DIR = os.path.join(BASE_DIR, "Data Set", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "Data Set", "models")
RESULTS_DIR = os.path.join(PROCESSED_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FLOWS_FILE = os.path.join(PROCESSED_DIR, "phase5_flows.csv")
GROUND_TRUTH_FILE = os.path.join(PROCESSED_DIR, "lab_traffic_ground_truth.csv")
TIER1_MODEL_FILE = os.path.join(MODELS_DIR, "tier1_fastpath_binary.json")
TIER3_MODEL_FILE = os.path.join(MODELS_DIR, "tier3_full_classifier_xgboost.json")

TIER1_RESULTS_FILE = os.path.join(RESULTS_DIR, "tier1_baseline_results.json")
TIER3_RESULTS_FILE = os.path.join(RESULTS_DIR, "tier3_baseline_results.json")

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

CLASS_MAPPING = {
    'BENIGN': 0,
    'DoS_DDoS': 1,
    'BRUTE_FORCE': 2,
    'INFILTRATION': 3,
    'BOTNET': 4,
    'WEB_ATTACK': 5
}
INV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}


def load_and_preprocess_flows():
    print(f"[*] Loading live extracted flows: {FLOWS_FILE}")
    flows_df = pd.read_csv(FLOWS_FILE)
    print(f"    Raw flows loaded: {len(flows_df)}")

    flows_df.rename(columns=COLUMN_MAPPING, inplace=True)

    for col in ['Flow Bytes/s', 'Flow Packets/s']:
        if col in flows_df.columns:
            flows_df[col] = flows_df[col].replace([np.inf, -np.inf], np.nan)
            flows_df[col] = flows_df[col].fillna(0)

    flows_df[GOLDEN_12] = flows_df[GOLDEN_12].fillna(0)

    time_cols_to_scale = ['Flow Duration', 'Flow IAT Mean', 'Fwd IAT Max', 'Bwd IAT Min']
    print("[*] Scaling time features from seconds to microseconds (10^6)...")
    for col in time_cols_to_scale:
        if col in flows_df.columns:
            flows_df[col] = flows_df[col] * 1_000_000.0

    flows_df['parsed_time'] = pd.to_datetime(flows_df['timestamp'], errors='coerce', utc=True)
    return flows_df


def assign_ground_truth(flows_df):
    print(f"[*] Loading ground truth log: {GROUND_TRUTH_FILE}")
    gt_df = pd.read_csv(GROUND_TRUTH_FILE)
    gt_df['start'] = pd.to_datetime(gt_df['timestamp_start'], utc=True)
    gt_df['end'] = pd.to_datetime(gt_df['timestamp_end'], utc=True)

    gt_df = gt_df.sort_values('start').reset_index(drop=True)
    gt_df['time_diff'] = (gt_df['start'] - gt_df['end'].shift(1)).dt.total_seconds()
    gt_df['session_id'] = (gt_df['time_diff'] > 60).cumsum()

    session7 = gt_df[gt_df['session_id'] == 7].copy()

    flow_min_time = flows_df['parsed_time'].min()
    gt_min_time = session7['start'].min()
    time_offset = flow_min_time - gt_min_time

    session7['shifted_start'] = session7['start'] + time_offset
    session7['shifted_end'] = session7['end'] + time_offset

    flows_df['ground_truth_label'] = 'BENIGN'

    for _, attack in session7.iterrows():
        label = attack['class_label']
        if label == 'BENIGN':
            continue
        start, end = attack['shifted_start'], attack['shifted_end']
        mask = (flows_df['parsed_time'] >= start) & (flows_df['parsed_time'] <= end)
        if mask.sum() > 0:
            flows_df.loc[mask, 'ground_truth_label'] = label

    print("\n[*] Live Flow Ground-Truth Label Distribution:")
    print(flows_df['ground_truth_label'].value_counts().to_string())

    # Sanity check: warn if flows fall entirely outside the session7 window,
    # since those would silently default to BENIGN above.
    covered = (flows_df['parsed_time'] >= session7['shifted_start'].min()) & \
              (flows_df['parsed_time'] <= session7['shifted_end'].max())
    n_outside = (~covered).sum()
    if n_outside > 0:
        print(f"    [!] WARNING: {n_outside} flows fall outside the Session 7 ground-truth "
              f"window and were defaulted to BENIGN. Verify this is expected.")

    return flows_df


def evaluate_tier1(flows_df):
    print("\n" + "=" * 60)
    print("--- TIER 1: FAST-PATH BINARY CLASSIFIER EVALUATION (ZERO-SHOT BASELINE) ---")
    print("=" * 60)

    model = xgb.Booster()
    model.load_model(TIER1_MODEL_FILE)

    X = flows_df[GOLDEN_12]
    dmatrix = xgb.DMatrix(X)

    raw_preds = model.predict(dmatrix)
    preds = (raw_preds > 0.5).astype(int)

    y_true = (flows_df['ground_truth_label'] == 'DoS_DDoS').astype(int)

    unique_classes = sorted(list(set(y_true) | set(preds)))
    class_names_map = {0: 'Other/BENIGN', 1: 'DoS_DDoS'}
    target_names = [class_names_map[i] for i in unique_classes]

    report_text = classification_report(y_true, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0)
    report_dict = classification_report(y_true, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0, output_dict=True)

    print("\nClassification Report (Tier 1 - DoS_DDoS vs Other):")
    print(report_text)

    cm = confusion_matrix(y_true, preds, labels=unique_classes)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=[f"True_{class_names_map[i]}" for i in unique_classes], columns=[f"Pred_{class_names_map[i]}" for i in unique_classes]))

    dos_metrics = report_dict.get('DoS_DDoS', {})
    results = {
        'model_tag': 'zero_shot_baseline',
        'model_file': os.path.basename(TIER1_MODEL_FILE),
        'n_flows': int(len(flows_df)),
        'accuracy': report_dict.get('accuracy', 0.0),
        'dos_precision': dos_metrics.get('precision', 0.0),
        'dos_recall': dos_metrics.get('recall', 0.0),
        'dos_f1': dos_metrics.get('f1-score', 0.0),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': target_names,
        'full_classification_report': report_dict
    }
    return results


def evaluate_tier3(flows_df):
    print("\n" + "=" * 60)
    print("--- TIER 3: FULL MULTI-CLASS CLASSIFIER EVALUATION (ZERO-SHOT BASELINE) ---")
    print("=" * 60)

    model = xgb.Booster()
    model.load_model(TIER3_MODEL_FILE)

    X = flows_df[GOLDEN_12]
    dmatrix = xgb.DMatrix(X)

    raw_preds = model.predict(dmatrix)
    preds = np.argmax(raw_preds, axis=1) if len(raw_preds.shape) > 1 else (raw_preds > 0.5).astype(int)

    y_true = flows_df['ground_truth_label'].map(CLASS_MAPPING).fillna(0).astype(int)

    present_class_indices = sorted(list(set(y_true) | set(preds)))
    present_class_names = [INV_CLASS_MAPPING[i] for i in present_class_indices]

    report_text = classification_report(
        y_true, preds, labels=present_class_indices, target_names=present_class_names, digits=4, zero_division=0
    )
    report_dict = classification_report(
        y_true, preds, labels=present_class_indices, target_names=present_class_names, digits=4, zero_division=0, output_dict=True
    )

    print("\nClassification Report (Tier 3 - Full 6-Class):")
    print(report_text)

    cm = confusion_matrix(y_true, preds, labels=present_class_indices)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=[f"True_{c}" for c in present_class_names], columns=[f"Pred_{c}" for c in present_class_names]))

    results = {
        'model_tag': 'zero_shot_baseline',
        'model_file': os.path.basename(TIER3_MODEL_FILE),
        'n_flows': int(len(flows_df)),
        'accuracy': report_dict.get('accuracy', 0.0),
        'macro_f1': report_dict.get('macro avg', {}).get('f1-score', 0.0),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': present_class_names,
        'full_classification_report': report_dict
    }
    return results


def main():
    flows_df = load_and_preprocess_flows()
    flows_df = assign_ground_truth(flows_df)

    tier1_results = evaluate_tier1(flows_df)
    tier3_results = evaluate_tier3(flows_df)

    with open(TIER1_RESULTS_FILE, 'w') as f:
        json.dump(tier1_results, f, indent=4)
    with open(TIER3_RESULTS_FILE, 'w') as f:
        json.dump(tier3_results, f, indent=4)

    print(f"\n[+] Saved Tier 1 baseline results to: {TIER1_RESULTS_FILE}")
    print(f"[+] Saved Tier 3 baseline results to: {TIER3_RESULTS_FILE}")


if __name__ == "__main__":
    main()
