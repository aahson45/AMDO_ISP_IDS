"""
generate_lab_traffic.py
--------------------------
Phase 4: Generate a mix of BENIGN and attack-shaped traffic from Kali
toward Ubuntu-Victim, logging ground truth (exactly what was sent,
when, and what class it belongs to) so Phase 5 can score ADMO's real
classification accuracy against known-correct answers.

Run this FROM KALI. Requires: python3-requests, hping3, hydra, scapy
(all normally preinstalled on Kali; if missing: sudo apt install
hping3 hydra python3-scapy python3-requests)

Usage examples:
    sudo python3 generate_lab_traffic.py --benign --duration 60
    sudo python3 generate_lab_traffic.py --ddos --duration 20
    sudo python3 generate_lab_traffic.py --bruteforce
    sudo python3 generate_lab_traffic.py --webattack --duration 30
    sudo python3 generate_lab_traffic.py --botnet --duration 120
    sudo python3 generate_lab_traffic.py --all --duration 60

NOTE: --ddos and --bruteforce need root (raw sockets / hydra), hence
running the whole script with sudo is simplest.
"""

import argparse
import csv
import subprocess
import time
import random
import os
from datetime import datetime, timezone

# ============================================================
# 1. CONFIG
# ============================================================
TARGET_IP = "10.10.10.5"       # Ubuntu-Victim
WEB_PORT = 8080
SSH_PORT = 22

GROUND_TRUTH_CSV = os.path.expanduser("~/lab_traffic_ground_truth.csv")

# Disposable test account created on Ubuntu specifically for this
# brute-force scenario - never a real credential.
BRUTEFORCE_TEST_USER = "bftest"
BRUTEFORCE_TEST_PASSWORD_LIST = "/tmp/bf_passwords.txt"  # small wordlist we generate below

GROUND_TRUTH_FIELDS = ["timestamp_start", "timestamp_end", "class_label",
                        "src_ip", "dst_ip", "dst_port", "tool", "detail"]


def log_ground_truth(writer, start, end, label, dst_port, tool, detail=""):
    writer.writerow({
        "timestamp_start": start.isoformat(),
        "timestamp_end": end.isoformat(),
        "class_label": label,
        "src_ip": "10.10.20.5",  # Kali's own address - this machine
        "dst_ip": TARGET_IP,
        "dst_port": dst_port,
        "tool": tool,
        "detail": detail,
    })


def open_ground_truth_writer():
    file_exists = os.path.exists(GROUND_TRUTH_CSV)
    f = open(GROUND_TRUTH_CSV, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=GROUND_TRUTH_FIELDS)
    if not file_exists:
        writer.writeheader()
    return f, writer


# ============================================================
# 2. BENIGN - ordinary HTTP requests at random intervals
# ============================================================

def generate_benign(duration_sec, writer):
    import requests
    print(f"[BENIGN] Generating for {duration_sec}s...")
    end_time = time.time() + duration_sec
    while time.time() < end_time:
        start = datetime.now(timezone.utc)
        status = "SUCCESS"
        try:
            resp = requests.get(f"http://{TARGET_IP}:{WEB_PORT}/", timeout=2)
            status = f"SUCCESS (HTTP {resp.status_code})"
        except requests.exceptions.RequestException as e:
            status = f"FAILED ({type(e).__name__})"
            print(f"  (request failed, continuing: {e})")
        end = datetime.now(timezone.utc)
        log_ground_truth(writer, start, end, "BENIGN", WEB_PORT, "requests", f"GET / - {status}")
        time.sleep(random.uniform(0.5, 2.5))  # irregular, human-like spacing
    print("[BENIGN] Done.")


# ============================================================
# 3. DoS_DDoS - hping3 SYN flood, LAB-SAFE (bounded count, not unbounded)
# ============================================================

def generate_ddos(duration_sec, writer):
    print(f"[DoS_DDoS] Generating hping3 SYN flood for {duration_sec}s...")
    start = datetime.now(timezone.utc)

    # -S: SYN flag, -p: target port, -i u1000: send every 1000 microseconds
    # (1000 pps - deliberately bounded/lab-safe, not a maximum-rate flood),
    # -c: fixed packet COUNT so this can never run away unbounded even if
    # something goes wrong with timing.
    packet_count = duration_sec * 1000  # ~1000 packets/sec * duration
    cmd = [
        "hping3", "-S", "-p", str(WEB_PORT), "-i", "u1000",
        "-c", str(packet_count), TARGET_IP,
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    end = datetime.now(timezone.utc)
    log_ground_truth(writer, start, end, "DoS_DDoS", WEB_PORT, "hping3",
                      f"SYN flood, ~{packet_count} packets, ~1000pps (bounded, lab-safe rate)")
    print("[DoS_DDoS] Done.")


# ============================================================
# 4. BRUTE_FORCE - hydra against the disposable test SSH account
# ============================================================

def generate_bruteforce(writer):
    print("[BRUTE_FORCE] Generating hydra SSH brute-force attempt...")

    # Small wordlist: mostly wrong guesses, with the real disposable
    # password included so the attempt set is realistic (a real
    # brute-forcer doesn't know the password in advance either).
    wordlist = ["123456", "password", "admin123", "letmein",
                "qwerty123", "weakpass123", "test1234", "changeme"]
    with open(BRUTEFORCE_TEST_PASSWORD_LIST, "w") as f:
        f.write("\n".join(wordlist))

    start = datetime.now(timezone.utc)
    cmd = [
        "hydra", "-l", BRUTEFORCE_TEST_USER, "-P", BRUTEFORCE_TEST_PASSWORD_LIST,
        "-t", "4",  # 4 parallel attempts - modest, lab-scale, not maximally aggressive
        "-f",       # stop after first successful login (this is a test account, expected to succeed)
        f"ssh://{TARGET_IP}:{SSH_PORT}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = datetime.now(timezone.utc)

    log_ground_truth(writer, start, end, "BRUTE_FORCE", SSH_PORT, "hydra",
                      f"{len(wordlist)} password attempts against disposable test account")
    print("[BRUTE_FORCE] Done.")
    print(result.stdout[-500:])  # last bit of hydra's own output, for a quick sanity glance


# ============================================================
# 5. WEB_ATTACK - scapy-crafted HTTP requests with SQLi/XSS-shaped payloads
# ============================================================

def generate_webattack(duration_sec, writer):
    import requests
    print(f"[WEB_ATTACK] Generating SQLi/XSS-shaped requests for {duration_sec}s...")

    # Pattern-shape only, per project scope - these mimic the STRUCTURE
    # of real attack payloads (for flow-feature realism: unusual URL
    # length, special characters) without targeting any real service.
    payloads = [
        "/login?user=admin' OR '1'='1&pass=x",
        "/search?q=<script>alert(1)</script>",
        "/product?id=1; DROP TABLE users;--",
        "/comment?text=<img src=x onerror=alert(1)>",
    ]

    end_time = time.time() + duration_sec
    while time.time() < end_time:
        payload = random.choice(payloads)
        start = datetime.now(timezone.utc)
        try:
            requests.get(f"http://{TARGET_IP}:{WEB_PORT}{payload}", timeout=2)
        except requests.exceptions.RequestException as e:
            print(f"  (request failed, continuing: {e})")
        end = datetime.now(timezone.utc)
        log_ground_truth(writer, start, end, "WEB_ATTACK", WEB_PORT, "requests", payload)
        time.sleep(random.uniform(1.0, 3.0))
    print("[WEB_ATTACK] Done.")


# ============================================================
# 6. BOTNET - periodic low-volume beacon (C2 heartbeat pattern)
# ============================================================

def generate_botnet(duration_sec, writer):
    import requests
    print(f"[BOTNET] Generating periodic beacon traffic for {duration_sec}s...")
    end_time = time.time() + duration_sec
    beacon_interval = 15  # fixed interval - the regularity IS the signature

    while time.time() < end_time:
        start = datetime.now(timezone.utc)
        try:
            requests.get(f"http://{TARGET_IP}:{WEB_PORT}/", timeout=2,
                         headers={"User-Agent": "beacon-check/1.0"})
        except requests.exceptions.RequestException as e:
            print(f"  (beacon failed, continuing: {e})")
        end = datetime.now(timezone.utc)
        log_ground_truth(writer, start, end, "BOTNET", WEB_PORT, "requests",
                          f"fixed-interval beacon ({beacon_interval}s)")
        time.sleep(beacon_interval)  # FIXED interval, not randomized - unlike benign traffic
    print("[BOTNET] Done.")


# ============================================================
# 7. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ADMO Phase 4 lab traffic generator")
    parser.add_argument("--benign", action="store_true")
    parser.add_argument("--ddos", action="store_true")
    parser.add_argument("--bruteforce", action="store_true")
    parser.add_argument("--webattack", action="store_true")
    parser.add_argument("--botnet", action="store_true")
    parser.add_argument("--all", action="store_true", help="run every class in sequence")
    parser.add_argument("--duration", type=int, default=60,
                         help="duration in seconds for time-based classes (benign/ddos/webattack/botnet)")
    args = parser.parse_args()

    f, writer = open_ground_truth_writer()

    try:
        if args.benign or args.all:
            generate_benign(args.duration, writer)
        if args.ddos or args.all:
            generate_ddos(min(args.duration, 20), writer)  # capped - flood traffic, keep it brief
        if args.bruteforce or args.all:
            generate_bruteforce(writer)
        if args.webattack or args.all:
            generate_webattack(args.duration, writer)
        if args.botnet or args.all:
            generate_botnet(args.duration, writer)
    finally:
        f.close()

    print(f"\nGround truth log written to: {GROUND_TRUTH_CSV}")


if __name__ == "__main__":
    main()
