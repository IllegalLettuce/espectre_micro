import paho.mqtt.client as mqtt
import json
import time
import threading
import numpy as np
import joblib
import csv
from collections import deque
from datetime import datetime
import base64
import struct

# ── Configuration ─────────────────────────────────────────────────────────────
GROUND_TRUTH       = "Quadrant_1"
WARMUP_SECONDS     = 3
COLLECTION_SECONDS = 45

MODEL_PATH  = 'rf_spatial_classifier.pkl'
SCALER_PATH = 'rf_scaler.pkl'


# ── Model ─────────────────────────────────────────────────────────────────────
model       = joblib.load(MODEL_PATH)
scaler      = joblib.load(SCALER_PATH)
LABEL_MAP   = {0: 'Baseline', 1: 'Quadrant_1', 2: 'Quadrant_2'}
CLASS_NAMES = ['Baseline', 'Quadrant_1', 'Quadrant_2']

print(f"Model loaded:  {MODEL_PATH}")
print(f"Classes: {CLASS_NAMES}")
print(f"Ground truth: '{GROUND_TRUTH}'")
print(f"Model classes (raw): {model.classes_}")
if GROUND_TRUTH not in CLASS_NAMES:
    raise ValueError(f"GROUND_TRUTH '{GROUND_TRUTH}' not in {CLASS_NAMES}")

# ── Feature extraction ────────────────────────────────────────────────────────
SEQ_LEN  = 28
VALID_SC = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
            21,22,23,24,25,26,34,35,36,37,38,39,40,41,42,43]
# ── CSV ────────────────────────────────────────────────────────
NUM_SC_AMPS = 31
SC_AMP_COLS = [f'sc_amp_{i}' for i in VALID_SC]

csv_filename = f'csi_eval_{GROUND_TRUTH}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
csv_file     = open(csv_filename, 'a', newline='')
csv_writer   = csv.writer(csv_file)

csv_writer.writerow([
    'timestamp',
    'entropy_turb', 'iqr_turb', 'variance_turb',
    'skewness', 'kurtosis',
    'amp_mean', 'amp_range', 'amp_std',
    'amp_mean_low', 'amp_mean_mid', 'amp_mean_high',
    *SC_AMP_COLS,
    'phase_mean', 'phase_std', 'phase_range',
    'ground_truth', 'predicted', 'correct',
    'conf_baseline', 'conf_quadrant1', 'conf_quadrant2', 'confidence',
    'inference_ms',
    'mqtt_transport_ms',   # ESP32 publish → laptop receipt
    'payload_bytes',
    'esp32_pps',
    'packets_dropped',
])
print(f"CSV: {csv_filename}\n")


AGG_FEATURES = [
    "entropy_turb", "iqr_turb", "variance_turb",
    "skewness", "kurtosis",
    "amp_mean", "amp_range", "amp_std",
    "amp_mean_low", "amp_mean_mid", "amp_mean_high",
    "phase_diff_mean", "phase_diff_std",
    "phase_diff_range", "phase_diff_skew",
]



def extract_window_features(agg_buf, sc_buf):
    agg_arr = np.array([[row[f] for f in AGG_FEATURES] for row in agg_buf], dtype='float32')
    sc_arr = np.array([frame for frame in sc_buf], dtype='float32')


    feats = []
    for i in range(agg_arr.shape[1]):
        col = agg_arr[:, i]
        feats += [float(np.mean(col)), float(np.std(col)), float(np.max(col)),
                  float(np.percentile(col, 25)), float(np.percentile(col, 75))]

    feats.extend(np.mean(sc_arr, axis=0).tolist())
    feats.extend(np.std(sc_arr,  axis=0).tolist())

    vt      = agg_arr[:, 2]
    vt_mean = float(np.mean(vt))
    vt_max  = float(np.max(vt))
    vt_std  = float(np.std(vt))
    feats  += [vt_max, vt_std, float(np.sum(vt > vt_mean + 2*vt_std)),
               float(vt_max / (vt_mean + 1e-6))]

    half = SEQ_LEN // 2
    feats.append(float(np.mean(vt[half:]) - np.mean(vt[:half])))

    return np.array(feats, dtype='float32')


# ── State ─────────────────────────────────────────────────────────────────────
agg_buffer  = deque(maxlen=SEQ_LEN)
sc_buffer   = deque(maxlen=SEQ_LEN)              

warmup_complete   = False
collection_active = False
collection_start  = None

results      = []
pred_counts  = {c: 0 for c in CLASS_NAMES}
latency_log  = []        # inference_ms
transport_log = []       # mqtt_transport_ms
traffic_log  = []        # payload bytes
msg_times    = []        # wall-clock arrival time of every message
pps_log      = []
dropped_log  = []

# Clock offset — computed once from first message
clock_offset_ms = None   # wall_clock_ms - esp32_boot_ms


# ── Timers ────────────────────────────────────────────────────────────────────
def start_collection():
    global warmup_complete, collection_active, collection_start
    warmup_complete   = True
    collection_active = True
    collection_start  = time.time()
    print(f"\n{'='*65}")
    print(f"  COLLECTING — {GROUND_TRUTH}  ({COLLECTION_SECONDS}s)")
    print(f"{'='*65}\n")
    threading.Timer(COLLECTION_SECONDS, stop_collection).start()


def stop_collection():
    global collection_active
    collection_active = False

    total    = len(results)
    correct  = sum(1 for _, _, ok, *_ in results if ok)
    accuracy = correct / total * 100 if total > 0 else 0

    # ── Traffic ───────────────────────────────────────────────────────────
    if traffic_log and msg_times:
        avg_bytes  = np.mean(traffic_log)
        duration   = msg_times[-1] - msg_times[0] if len(msg_times) > 1 else COLLECTION_SECONDS
        actual_hz  = len(msg_times) / duration
        kbps       = (avg_bytes * actual_hz) / 1024

        print(f"\n{'='*65}")
        print(f"  TRAFFIC MEASUREMENT  (NFR5 — must be < 25 KB/s)")
        print(f"{'='*65}")
        print(f"  Messages received  : {len(traffic_log)}")
        print(f"  Avg payload size   : {avg_bytes:.0f} bytes")
        print(f"  Min / Max payload  : {min(traffic_log)} / {max(traffic_log)} bytes")
        print(f"  Actual message rate: {actual_hz:.1f} Hz")
        print(f"  Estimated traffic  : {kbps:.2f} KB/s")
        if pps_log:
            print(f"  ESP32 avg pps      : {np.mean(pps_log):.1f}")
        if dropped_log:
            print(f"  Avg packets dropped: {np.mean(dropped_log):.2f}")
        print(f"  NFR5 {'✓ PASS' if kbps < 25 else '✗ FAIL'}  (threshold: 25 KB/s)")

    # ── MQTT transport latency ────────────────────────────────────────────
    if transport_log:
        print(f"\n{'='*65}")
        print(f"  MQTT TRANSPORT LATENCY  (ESP32 publish → laptop receipt)")
        print(f"{'='*65}")
        print(f"  Samples            : {len(transport_log)}")
        print(f"  Avg transport      : {np.mean(transport_log):.2f} ms")
        print(f"  Min / Max          : {np.min(transport_log):.2f} / {np.max(transport_log):.2f} ms")
        print(f"  p95                : {np.percentile(transport_log, 95):.2f} ms")
        print(f"  Note: clock offset established from first message — "
              f"valid as long as ESP32 did not reboot during test")

    # ── Inference latency ─────────────────────────────────────────────────
    if latency_log:
        print(f"\n{'='*65}")
        print(f"  INFERENCE LATENCY  (FR2/NFR2 — must be < 1000ms total)")
        print(f"{'='*65}")
        print(f"  Predictions made   : {len(latency_log)}")
        print(f"  Avg inference      : {np.mean(latency_log):.2f} ms")
        print(f"  Max inference      : {np.max(latency_log):.2f} ms")
        print(f"  p95 inference      : {np.percentile(latency_log, 95):.2f} ms")

        if transport_log:
            combined = np.mean(transport_log) + np.mean(latency_log)
            print(f"\n  Combined (transport + inference): {combined:.2f} ms")
            print(f"  FR2/NFR2 {'✓ PASS' if combined < 1000 else '✗ FAIL'}  (threshold: 1000ms)")

    # ── Classification ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  CLASSIFICATION RESULTS")
    print(f"{'='*65}")
    print(f"  Ground truth : {GROUND_TRUTH}")
    print(f"  CSV output   : {csv_filename}")
    print(f"  Total preds  : {total}")
    print(f"  Correct      : {correct}")
    print(f"  Accuracy     : {accuracy:.1f}%")

    print(f"\n  Prediction breakdown:")
    for cls in CLASS_NAMES:
        count  = pred_counts[cls]
        pct    = count / total * 100 if total > 0 else 0
        bar    = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
        marker = ' ← ground truth' if cls == GROUND_TRUTH else ''
        print(f"    {cls:12s}: {count:4d} ({pct:5.1f}%) [{bar}]{marker}")

    correct_conf   = [c for _, c, ok, *_ in results if ok]
    incorrect_conf = [c for _, c, ok, *_ in results if not ok]
    if correct_conf:
        print(f"\n  Avg confidence (correct)  : {np.mean(correct_conf):.3f}")
    if incorrect_conf:
        print(f"  Avg confidence (incorrect): {np.mean(incorrect_conf):.3f}")

    print(f"{'='*65}\n")
    csv_file.close()
    client.disconnect()
    client.loop_stop()


# ── MQTT ──────────────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT broker")
        client.subscribe("home/espectre/node1")
        print(f"Subscribed — warmup {WARMUP_SECONDS}s...")
        threading.Timer(WARMUP_SECONDS, start_collection).start()

def on_message(client, userdata, msg):
    global clock_offset_ms

    recv_wall_ms  = time.time() * 1000
    payload_bytes = len(msg.payload)
    traffic_log.append(payload_bytes)
    msg_times.append(time.time())

    try:
        payload = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        return

    if 'features' not in payload:
        return

    pps_log.append(payload.get('pps', 0))
    dropped_log.append(payload.get('packets_dropped', 0))

    if not warmup_complete:
        return

    f       = payload['features']
    sc_raw = f.get('sc_amps', '')
    if isinstance(sc_raw, str) and sc_raw:
        buf = base64.b64decode(sc_raw)
        decoded = [struct.unpack_from('>H', buf, i * 2)[0] / 100.0
                for i in range(len(buf) // 2)]
    else:
        decoded = sc_raw if isinstance(sc_raw, list) else []

    sc_amps = [float(decoded[i]) if i < len(decoded) else 0.0 for i in VALID_SC]


    agg_buffer.append({feat: f.get(feat, 0.0) for feat in AGG_FEATURES})
    sc_buffer.append(sc_amps)

    if not collection_active:
        return

    if len(agg_buffer) < SEQ_LEN:
        elapsed = time.time() - collection_start if collection_start else 0
        print(f"\r  Buffering {len(agg_buffer)}/{SEQ_LEN} | {elapsed:.1f}s", end='', flush=True)
        return

    # ── Inference ─────────────────────────────────────────────────────────
    t0              = time.time()
    features        = extract_window_features(list(agg_buffer), list(sc_buffer))
    features_scaled = scaler.transform(features.reshape(1, -1))
    proba           = model.predict_proba(features_scaled)[0]
    pipeline_ms     = (time.time() - t0) * 1000   

    pred_idx = int(np.argmax(proba))
    pred     = LABEL_MAP[pred_idx]
    conf     = float(proba[pred_idx])
    correct  = pred == GROUND_TRUTH

    prob_map = {LABEL_MAP[i]: float(proba[i]) for i in range(len(proba))}
    latency_log.append(pipeline_ms)
    results.append((pred, conf, correct, proba))
    pred_counts[pred] += 1

    latest = list(agg_buffer)[-1]
    csv_writer.writerow([
        datetime.now().isoformat(),
        latest.get('entropy_turb', 0), latest.get('iqr_turb', 0),
        latest.get('variance_turb', 0), latest.get('skewness', 0),
        latest.get('kurtosis', 0), latest.get('amp_mean', 0),
        latest.get('amp_range', 0), latest.get('amp_std', 0),
        latest.get('amp_mean_low', 0), latest.get('amp_mean_mid', 0),
        latest.get('amp_mean_high', 0),
        *sc_amps,
        f.get('phase_mean', 0), f.get('phase_std', 0), f.get('phase_range', 0),
        GROUND_TRUTH, pred, correct,
        round(prob_map.get('Baseline',   0), 4),
        round(prob_map.get('Quadrant_1', 0), 4),
        round(prob_map.get('Quadrant_2', 0), 4),
        round(conf, 4),
        round(pipeline_ms, 3),
        '',                              
        payload_bytes,
        payload.get('pps', ''),
        payload.get('packets_dropped', ''),
    ])
    csv_file.flush()

    elapsed     = time.time() - collection_start
    remaining   = max(0, COLLECTION_SECONDS - elapsed)
    running_acc = sum(1 for _, _, ok, *_ in results if ok) / len(results) * 100

    COLOURS = {'Baseline': '\033[92m', 'Quadrant_1': '\033[93m', 'Quadrant_2': '\033[91m'}
    RESET   = '\033[0m'
    colour  = COLOURS.get(pred, '')

    print(
        f"{colour}{pred:12s}{RESET} conf={conf:.2f} | "
        f"B={prob_map['Baseline']:.2f} Q1={prob_map['Quadrant_1']:.2f} "
        f"Q2={prob_map['Quadrant_2']:.2f} | "
        f"{remaining:.0f}s left | acc={running_acc:.1f}% | "
        f"infer={pipeline_ms:.1f}ms | {payload_bytes}B"   
    )



# ── Entry point ───────────────────────────────────────────────────────────────
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.username_pw_set("mqtt_device", "cheese")
client.on_connect = on_connect
client.on_message = on_message

print(f"\nConnecting to 192.168.3.115:1883...")
client.connect("192.168.3.115", 1883, 60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nInterrupted.")
    csv_file.close()
    client.disconnect()
