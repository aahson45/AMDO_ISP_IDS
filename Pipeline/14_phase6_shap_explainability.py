import os
import numpy as np
import pandas as pd
import xgboost as xgb

# --- PATH CONFIGURATION ---
BASE_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software"
PROCESSED_DIR = os.path.join(BASE_DIR, "Data Set", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "Data Set", "models")

LIVE_FLOWS_FILE = os.path.join(PROCESSED_DIR, "phase5_flows.csv")
BASELINE_MODEL_FILE = os.path.join(MODELS_DIR, "tier1_fastpath_binary.json")
ADAPTED_MODEL_FILE = os.path.join(MODELS_DIR, "tier1_admo_adapted.json")
SHAP_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "shap_explainability_summary.csv")

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


def load_and_preprocess_live_data():
    print(f"[*] Ingesting live telemetry flows: {LIVE_FLOWS_FILE}")
    df = pd.read_csv(LIVE_FLOWS_FILE)
    df.rename(columns=COLUMN_MAPPING, inplace=True)

    for col in ['Flow Bytes/s', 'Flow Packets/s']:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    df[GOLDEN_12] = df[GOLDEN_12].fillna(0)

    # Scale time metrics to microseconds (10^6)
    time_cols = ['Flow Duration', 'Flow IAT Mean', 'Fwd IAT Max', 'Bwd IAT Min']
    for col in time_cols:
        df[col] = df[col] * 1_000_000.0

    return df


def run_shap_analysis():
    df_live = load_and_preprocess_live_data()
    X = df_live[GOLDEN_12]
    dmatrix = xgb.DMatrix(X)

    # 1. Load Baseline and ADMO Models
    print(f"[*] Loading Baseline Model: {BASELINE_MODEL_FILE}")
    baseline_model = xgb.Booster()
    baseline_model.load_model(BASELINE_MODEL_FILE)

    print(f"[*] Loading ADMO Adapted Model: {ADAPTED_MODEL_FILE}")
    adapted_model = xgb.Booster()
    adapted_model.load_model(ADAPTED_MODEL_FILE)

    # 2. Compute Native TreeSHAP Contributions
    print("\n[+] Computing Native TreeSHAP Feature Contributions on Live DoS Traffic...")
    
    # SHAP returns matrix of shape (N, num_features + 1), where last column is bias term
    shap_baseline = baseline_model.predict(dmatrix, pred_contribs=True)[:, :-1]
    shap_adapted = adapted_model.predict(dmatrix, pred_contribs=True)[:, :-1]

    # Calculate Mean Absolute SHAP values per feature
    mean_abs_shap_base = np.mean(np.abs(shap_baseline), axis=0)
    mean_abs_shap_adapt = np.mean(np.abs(shap_adapted), axis=0)

    # 3. Create Comparison Table
    summary_df = pd.DataFrame({
        'Feature': GOLDEN_12,
        'Baseline_Mean_Abs_SHAP': mean_abs_shap_base,
        'ADMO_Adapted_Mean_Abs_SHAP': mean_abs_shap_adapt
    })

    summary_df['SHAP_Shift_Delta'] = summary_df['ADMO_Adapted_Mean_Abs_SHAP'] - summary_df['Baseline_Mean_Abs_SHAP']
    summary_df = summary_df.sort_values(by='ADMO_Adapted_Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("--- SHAP FEATURE IMPORTANCE COMPARISON (Baseline vs ADMO) ---")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Save to CSV
    summary_df.to_csv(SHAP_OUTPUT_FILE, index=False)
    print(f"\n[+] Saved SHAP explainability analysis to: {SHAP_OUTPUT_FILE}")

    # Top Feature Highlights
    top_base = summary_df.sort_values(by='Baseline_Mean_Abs_SHAP', ascending=False).iloc[0]['Feature']
    top_adapt = summary_df.iloc[0]['Feature']
    
    print("\n=== XAI THESIS INSIGHT ===")
    print(f"• Baseline Model Primary Driver  : '{top_base}'")
    print(f"• ADMO Adapted Primary Driver   : '{top_adapt}'")


if __name__ == "__main__":
    run_shap_analysis()