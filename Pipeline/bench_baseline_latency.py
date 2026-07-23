"""
bench_baseline_latency.py
----------------------------
Phase 3: Type B baseline latency measurement.

Reads two tcpdump captures taken SIMULTANEOUSLY on pfSense's own
WAN (em0) and LAN (em1) interfaces - same machine, same clock, so
there's no cross-machine clock-sync error to worry about.

For each ICMP echo request, matches its appearance in wan_capture.pcap
(when it arrived at pfSense) against its appearance in lan_capture.pcap
(when it left toward Ubuntu). The difference between those two
timestamps, for the SAME packet, is the actual forwarding latency -
this is the real "Type B" measurement your thesis's core claim rests on.

Requires: scapy (for reading pcap files and parsing ICMP fields)

Input:  wan_capture.pcap, lan_capture.pcap
Output: a per-packet latency CSV + a summary report (mean/median/p95/etc.)
"""

from scapy.all import rdpcap, ICMP, IP
import pandas as pd
import numpy as np
import os

# ============================================================
# 1. CONFIG
# ============================================================
DATA_DIR = r"C:\Users\usid3\Downloads\Ali Ahson Thesis Software\Data Set"
WAN_PCAP = os.path.join(DATA_DIR, "wan_capture.pcap")
LAN_PCAP = os.path.join(DATA_DIR, "lan_capture.pcap")

REPORT_DIR = os.path.join(DATA_DIR, "report")
OUTPUT_CSV = os.path.join(REPORT_DIR, "baseline_latency_per_packet.csv")
OUTPUT_REPORT = os.path.join(REPORT_DIR, "phase3_baseline_latency_report.txt")


def extract_icmp_echo_requests(pcap_path, report_lines, label):
    """
    Reads a pcap file and returns a dict mapping (icmp_id, icmp_seq) ->
    timestamp, but ONLY for ICMP echo REQUEST packets (type 8) -
    matching just the request half of each ping avoids accidentally
    pairing a request in one file with a reply in the other, which
    would silently produce nonsense timings.
    """
    packets = rdpcap(pcap_path)
    report_lines.append(f"{label}: {len(packets)} total packets in capture")

    request_times = {}
    reply_count = 0
    other_count = 0

    for pkt in packets:
        if ICMP in pkt:
            icmp_layer = pkt[ICMP]
            if icmp_layer.type == 8:  # echo request
                key = (icmp_layer.id, icmp_layer.seq)
                # pkt.time is a float Unix timestamp with microsecond
                # precision, captured by tcpdump at the moment it saw
                # the packet - this precision is exactly why tcpdump,
                # not ping, is the right tool for this measurement.
                request_times[key] = float(pkt.time)
            elif icmp_layer.type == 0:  # echo reply
                reply_count += 1
            else:
                other_count += 1

    report_lines.append(f"{label}: {len(request_times)} echo REQUESTS, "
                          f"{reply_count} echo REPLIES (ignored), {other_count} other ICMP (ignored)")
    return request_times


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_lines = ["ADMO THESIS - PHASE 3: TYPE B BASELINE LATENCY REPORT"]
    report_lines.append("(No IDS, no ADMO running - pure pfSense forwarding baseline)")

    wan_requests = extract_icmp_echo_requests(WAN_PCAP, report_lines, "WAN capture")
    lan_requests = extract_icmp_echo_requests(LAN_PCAP, report_lines, "LAN capture")

    # --- Match packets present in BOTH captures ---
    common_keys = set(wan_requests.keys()) & set(lan_requests.keys())
    wan_only = set(wan_requests.keys()) - set(lan_requests.keys())
    lan_only = set(lan_requests.keys()) - set(wan_requests.keys())

    report_lines.append(f"\nMatched (present in both captures): {len(common_keys)}")
    report_lines.append(f"WAN-only (never made it to LAN - dropped or blocked): {len(wan_only)}")
    report_lines.append(f"LAN-only (should not be possible - would indicate a capture artifact): {len(lan_only)}")

    if len(common_keys) == 0:
        report_lines.append("\nERROR: no matching packets found between the two captures. "
                              "Cannot compute latency. Check that both captures were taken "
                              "during the same ping burst and that ICMP id/seq fields are intact.")
        print("\n".join(report_lines))
        return

    # --- Compute per-packet latency: LAN timestamp - WAN timestamp ---
    rows = []
    for key in sorted(common_keys):
        icmp_id, icmp_seq = key
        wan_ts = wan_requests[key]
        lan_ts = lan_requests[key]
        latency_ms = (lan_ts - wan_ts) * 1000  # seconds -> milliseconds
        rows.append({
            "icmp_id": icmp_id,
            "icmp_seq": icmp_seq,
            "wan_timestamp": wan_ts,
            "lan_timestamp": lan_ts,
            "latency_ms": latency_ms,
        })

    df = pd.DataFrame(rows).sort_values("icmp_seq").reset_index(drop=True)

    # --- Sanity check: latency should never be negative ---
    # (a negative value would mean the packet appeared on LAN BEFORE
    # WAN, which is physically impossible for correctly forwarded
    # traffic and would indicate a clock or matching problem)
    negative_count = (df["latency_ms"] < 0).sum()
    if negative_count > 0:
        report_lines.append(f"\nWARNING: {negative_count} packets show NEGATIVE latency - "
                              "this should be physically impossible and suggests a matching "
                              "or capture issue. Review these rows before trusting the results.")

    df.to_csv(OUTPUT_CSV, index=False)

    # --- Summary statistics ---
    report_lines.append(f"\n--- BASELINE FORWARDING LATENCY (WAN-in to LAN-out), N={len(df)} packets ---")
    report_lines.append(f"Mean:   {df['latency_ms'].mean():.4f} ms")
    report_lines.append(f"Median: {df['latency_ms'].median():.4f} ms")
    report_lines.append(f"Min:    {df['latency_ms'].min():.4f} ms")
    report_lines.append(f"P95:    {df['latency_ms'].quantile(0.95):.4f} ms")
    report_lines.append(f"P99:    {df['latency_ms'].quantile(0.99):.4f} ms")
    report_lines.append(f"Max:    {df['latency_ms'].max():.4f} ms")
    report_lines.append(f"Std dev:{df['latency_ms'].std():.4f} ms")

    report_lines.append(f"\nThis is the TRUE baseline forwarding latency with NO IDS and NO ADMO "
                          f"running - the reference point every later Phase 5 comparison measures against.")
    report_lines.append(f"\nPer-packet data saved to: {OUTPUT_CSV}")

    report_text = "\n".join(report_lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)
    print(f"\nReport saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
