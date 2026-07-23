import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- PATH CONFIGURATION ---
BASE_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software"
SPLITS_DIR = os.path.join(BASE_DIR, "Data Set", "splits")
PROCESSED_DIR = os.path.join(BASE_DIR, "Data Set", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "Data Set", "models")
RESULTS_DIR = os.path.join(PROCESSED_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BENCHMARK_TRAIN = os.path.join(SPLITS_DIR, "train.csv")
FLOWS_FILE = os.path.join(PROCESSED_DIR, "phase5_flows.csv")
GROUND_TRUTH_FILE = os.path.join(PROCESSED_DIR, "lab_traffic_ground_truth.csv")

TIER1_ADAPTED_MODEL = os.path.join(MODELS_DIR, "tier1_admo_adapted.json")
TIER3_ADAPTED_MODEL = os.path.join(MODELS_DIR, "tier3_admo_adapted.json")

TIER1_RESULTS_FILE = os.path.join(RESULTS_DIR, "tier1_admo_results.json")
TIER3_RESULTS_FILE = os.path.join(RESULTS_DIR, "tier3_admo_results.json")

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


def load_and_preprocess_live_flows():
    print(f"[*] Loading live extracted flows: {FLOWS_FILE}")
    flows_df = pd.read_csv(FLOWS_FILE)
    flows_df.rename(columns=COLUMN_MAPPING, inplace=True)

    for col in ['Flow Bytes/s', 'Flow Packets/s']:
        if col in flows_df.columns:
            flows_df[col] = flows_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    flows_df[GOLDEN_12] = flows_df[GOLDEN_12].fillna(0)

    time_cols = ['Flow Duration', 'Flow IAT Mean', 'Fwd IAT Max', 'Bwd IAT Min']
    for col in time_cols:
        flows_df[col] = flows_df[col] * 1_000_000.0

    flows_df['parsed_time'] = pd.to_datetime(flows_df['timestamp'], errors='coerce', utc=True)

    gt_df = pd.read_csv(GROUND_TRUTH_FILE)
    gt_df['start'] = pd.to_datetime(gt_df['timestamp_start'], utc=True)
    gt_df['end'] = pd.to_datetime(gt_df['timestamp_end'], utc=True)

    gt_df = gt_df.sort_values('start').reset_index(drop=True)
    gt_df['time_diff'] = (gt_df['start'] - gt_df['end'].shift(1)).dt.total_seconds()
    gt_df['session_id'] = (gt_df['time_diff'] > 60).cumsum()

    session7 = gt_df[gt_df['session_id'] == 7].copy()
    time_offset = flows_df['parsed_time'].min() - session7['start'].min()

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

    return flows_df


def run_admo_adaptation():
    live_df = load_and_preprocess_live_flows()

    live_train, live_test = train_test_split(
        live_df,
        test_size=0.70,
        random_state=42,
        stratify=live_df['ground_truth_label']
    )

    print(f"[*] Live Flows Split -> Adaptation Training: {len(live_train)} | Held-Out Test: {len(live_test)}")

    print(f"[*] Loading benchmark training dataset: {BENCHMARK_TRAIN}")
    bench_df = pd.read_csv(BENCHMARK_TRAIN)

    label_col = [c for c in bench_df.columns if 'label' in c.lower() or 'target' in c.lower()][0]

    if bench_df[label_col].dtype == object:
        bench_df['class_code'] = bench_df[label_col].map(CLASS_MAPPING).fillna(0).astype(int)
    else:
        bench_df['class_code'] = bench_df[label_col].astype(int)

    live_train_X = live_train[GOLDEN_12]
    live_train_y_tier3 = live_train['ground_truth_label'].map(CLASS_MAPPING).fillna(0).astype(int)
    live_train_y_tier1 = (live_train['ground_truth_label'] == 'DoS_DDoS').astype(int)

    bench_train_X = bench_df[GOLDEN_12]
    bench_train_y_tier3 = bench_df['class_code']
    bench_train_y_tier1 = (bench_df['class_code'] == 1).astype(int)

    X_train_combined = pd.concat([bench_train_X, live_train_X], ignore_index=True)
    y1_train_combined = pd.concat([bench_train_y_tier1, live_train_y_tier1], ignore_index=True)
    y3_train_combined = pd.concat([bench_train_y_tier3, live_train_y_tier3], ignore_index=True)

    print(f"[*] Combined Training Set Built: {len(X_train_combined)} total flow records.")

    # --- RETRAIN TIER 1 (FAST-PATH BINARY) ---
    print("\n[+] Retraining Tier 1 Fast-Path Binary Classifier with ADMO Domain Adaptation...")
    dtrain_t1 = xgb.DMatrix(X_train_combined, label=y1_train_combined)
    params_t1 = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'eta': 0.1,
        'tree_method': 'hist'
    }
    model_t1 = xgb.train(params_t1, dtrain_t1, num_boost_round=100)
    model_t1.save_model(TIER1_ADAPTED_MODEL)
    print(f"    Saved adapted Tier 1 model to: {TIER1_ADAPTED_MODEL}")

    # --- RETRAIN TIER 3 (FULL MULTI-CLASS CLASSIFIER) ---
    print("\n[+] Retraining Tier 3 Full Multi-Class Classifier with ADMO Domain Adaptation...")
    dtrain_t3 = xgb.DMatrix(X_train_combined, label=y3_train_combined)
    params_t3 = {
        'objective': 'multi:softprob',
        'num_class': 6,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'eta': 0.1,
        'tree_method': 'hist'
    }
    model_t3 = xgb.train(params_t3, dtrain_t3, num_boost_round=100)
    model_t3.save_model(TIER3_ADAPTED_MODEL)
    print(f"    Saved adapted Tier 3 model to: {TIER3_ADAPTED_MODEL}")

    # --- EVALUATE ON HELD-OUT LIVE TEST FLOWS (70%) ---
    print("\n" + "=" * 60)
    print("--- ADMO EVALUATION ON HELD-OUT LIVE LAB TEST FLOWS ---")
    print("=" * 60)

    X_test_live = live_test[GOLDEN_12]
    dtest_live = xgb.DMatrix(X_test_live)

    # 1. Evaluate Adapted Tier 1
    t1_preds_raw = model_t1.predict(dtest_live)
    t1_preds = (t1_preds_raw > 0.5).astype(int)
    t1_y_true = (live_test['ground_truth_label'] == 'DoS_DDoS').astype(int)

    t1_report_text = classification_report(t1_y_true, t1_preds, target_names=['Other/BENIGN', 'DoS_DDoS'], digits=4)
    t1_report_dict = classification_report(t1_y_true, t1_preds, target_names=['Other/BENIGN', 'DoS_DDoS'], digits=4, output_dict=True)

    print("\n[Tier 1 Adapted - Fast-Path Binary Classification Report]:")
    print(t1_report_text)
    t1_cm = confusion_matrix(t1_y_true, t1_preds)
    print("Tier 1 Confusion Matrix:")
    print(pd.DataFrame(t1_cm, index=['True Other', 'True DoS'], columns=['Pred Other', 'Pred DoS']))

    # 2. Evaluate Adapted Tier 3
    t3_preds_raw = model_t3.predict(dtest_live)
    t3_preds = np.argmax(t3_preds_raw, axis=1)
    t3_y_true = live_test['ground_truth_label'].map(CLASS_MAPPING).fillna(0).astype(int)

    present_indices = sorted(list(set(t3_y_true) | set(t3_preds)))
    present_names = [INV_CLASS_MAPPING[i] for i in present_indices]

    t3_report_text = classification_report(t3_y_true, t3_preds, labels=present_indices, target_names=present_names, digits=4, zero_division=0)
    t3_report_dict = classification_report(t3_y_true, t3_preds, labels=present_indices, target_names=present_names, digits=4, zero_division=0, output_dict=True)

    print("\n[Tier 3 Adapted - Full Multi-Class Classification Report]:")
    print(t3_report_text)
    t3_cm = confusion_matrix(t3_y_true, t3_preds, labels=present_indices)
    print("Tier 3 Confusion Matrix:")
    print(pd.DataFrame(t3_cm, index=[f"True_{c}" for c in present_names], columns=[f"Pred_{c}" for c in present_names]))

    # --- SAVE RESULTS FOR DOWNSTREAM THESIS ARTIFACTS ---
    dos_metrics = t1_report_dict.get('DoS_DDoS', {})
    tier1_results = {
        'model_tag': 'admo_adapted',
        'n_train_combined': int(len(X_train_combined)),
        'n_live_train': int(len(live_train)),
        'n_live_test_holdout': int(len(live_test)),
        'accuracy': t1_report_dict.get('accuracy', 0.0),
        'dos_precision': dos_metrics.get('precision', 0.0),
        'dos_recall': dos_metrics.get('recall', 0.0),
        'dos_f1': dos_metrics.get('f1-score', 0.0),
        'confusion_matrix': t1_cm.tolist(),
        'confusion_matrix_labels': ['Other/BENIGN', 'DoS_DDoS'],
        'full_classification_report': t1_report_dict
    }

    tier3_results = {
        'model_tag': 'admo_adapted',
        'n_train_combined': int(len(X_train_combined)),
        'n_live_test_holdout': int(len(live_test)),
        'accuracy': t3_report_dict.get('accuracy', 0.0),
        'macro_f1': t3_report_dict.get('macro avg', {}).get('f1-score', 0.0),
        'confusion_matrix': t3_cm.tolist(),
        'confusion_matrix_labels': present_names,
        'full_classification_report': t3_report_dict
    }

    with open(TIER1_RESULTS_FILE, 'w') as f:
        json.dump(tier1_results, f, indent=4)
    with open(TIER3_RESULTS_FILE, 'w') as f:
        json.dump(tier3_results, f, indent=4)

    print(f"\n[+] Saved Tier 1 ADMO-adapted results to: {TIER1_RESULTS_FILE}")
    print(f"[+] Saved Tier 3 ADMO-adapted results to: {TIER3_RESULTS_FILE}")


if __name__ == "__main__":
    run_admo_adaptation()
