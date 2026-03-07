import paho.mqtt.client as mqtt
import json
import time
import threading
import numpy as np
import joblib
from collections import deque
from datetime import datetime


# ── Configuration ─────────────────────────────────────────────────────────────
# Set this to the actual location you are walking in before running
GROUND_TRUTH = "Quadrant_1"   # Options: "Baseline", "Quadrant_1", "Quadrant_2"

WARMUP_SECONDS     = 3
COLLECTION_SECONDS = 45

MODEL_PATH  = 'rf_spatial_classifier.pkl'
SCALER_PATH = 'rf_scaler.pkl'


# ── Model ─────────────────────────────────────────────────────────────────────
model       = joblib.load(MODEL_PATH)
scaler      = joblib.load(SCALER_PATH)
CLASS_NAMES = list(model.classes_)

print(f"✓ Model loaded:  {MODEL_PATH}")
print(f"✓ Classes: {CLASS_NAMES}")
print(f"✓ Ground truth set to: '{GROUND_TRUTH}'")
if GROUND_TRUTH not in CLASS_NAMES:
    raise ValueError(f"GROUND_TRUTH '{GROUND_TRUTH}' not in model classes {CLASS_NAMES}")


# ── Feature extraction (must match training exactly) ──────────────────────────
SEQ_LEN  = 28
NULL_SC  = {0,1,2,3,4,5,27,28,29,30,31,32,33,34,35,59,60,61,62,63}
VALID_SC = [i for i in range(64) if i not in NULL_SC]
NUM_SC_AMPS = 64

AGG_FEATURES = [
    'entropy_turb', 'iqr_turb', 'variance_turb',
    'skewness', 'kurtosis',
    'amp_mean', 'amp_range', 'amp_std',
    'amp_mean_low', 'amp_mean_mid', 'amp_mean_high',
]


def extract_window_features(agg_buf, sc_buf):
    agg_arr = np.array([[row[f] for f in AGG_FEATURES] for row in agg_buf], dtype='float32')
    sc_arr  = np.array([[frame[i] for i in VALID_SC]   for frame in sc_buf],  dtype='float32')

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

warmup_complete    = False
collection_active  = False
collection_start   = None

# Per-prediction results
results     = []   # list of (predicted, confidence, correct)
pred_counts = {c: 0 for c in CLASS_NAMES}


# ── Timers ────────────────────────────────────────────────────────────────────
def start_collection():
    global warmup_complete, collection_active, collection_start
    warmup_complete   = True
    collection_active = True
    collection_start  = time.time()
    print(f"\n{'='*65}")
    print(f"  COLLECTING — walk around: {GROUND_TRUTH}  ({COLLECTION_SECONDS}s)")
    print(f"{'='*65}\n")
    threading.Timer(COLLECTION_SECONDS, stop_collection).start()


def stop_collection():
    global collection_active
    collection_active = False

    total      = len(results)
    correct    = sum(1 for _, _, ok in results if ok)
    accuracy   = correct / total * 100 if total > 0 else 0

    print(f"\n{'='*65}")
    print(f"  TEST COMPLETE")
    print(f"{'='*65}")
    print(f"  Ground truth : {GROUND_TRUTH}")
    print(f"  Total preds  : {total}")
    print(f"  Correct      : {correct}")
    print(f"  Accuracy     : {accuracy:.1f}%")

    print(f"\n  Prediction breakdown:")
    for cls in CLASS_NAMES:
        count = pred_counts[cls]
        pct   = count / total * 100 if total > 0 else 0
        bar   = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
        marker = ' ← ground truth' if cls == GROUND_TRUTH else ''
        print(f"    {cls:12s}: {count:4d} ({pct:5.1f}%) [{bar}]{marker}")

    # Confidence stats for correct vs incorrect predictions
    correct_conf   = [c for _, c, ok in results if ok]
    incorrect_conf = [c for _, c, ok in results if not ok]
    if correct_conf:
        print(f"\n  Avg confidence (correct)  : {np.mean(correct_conf):.3f}")
    if incorrect_conf:
        print(f"  Avg confidence (incorrect): {np.mean(incorrect_conf):.3f}")

    print(f"{'='*65}\n")
    client.disconnect()
    client.loop_stop()


# ── MQTT ──────────────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✓ Connected to MQTT broker")
        client.subscribe("home/espectre/node1")
        print(f"✓ Subscribed — warmup {WARMUP_SECONDS}s...")
        threading.Timer(WARMUP_SECONDS, start_collection).start()


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        return

    if 'features' not in payload:
        return

    if not warmup_complete:
        return

    f       = payload['features']
    sc_raw  = f.get('sc_amps', [])
    sc_amps = sc_raw[:NUM_SC_AMPS] + [0.0] * (NUM_SC_AMPS - len(sc_raw))

    agg_buffer.append({feat: f.get(feat, 0.0) for feat in AGG_FEATURES})
    sc_buffer.append(sc_amps)

    if not collection_active:
        return

    if len(agg_buffer) < SEQ_LEN:
        elapsed = time.time() - collection_start if collection_start else 0
        print(f"\r  Buffering {len(agg_buffer)}/{SEQ_LEN} | {elapsed:.1f}s", end='', flush=True)
        return

    features        = extract_window_features(list(agg_buffer), list(sc_buffer))
    features_scaled = scaler.transform(features.reshape(1, -1))
    proba           = model.predict_proba(features_scaled)[0]
    pred_idx        = int(np.argmax(proba))
    pred            = CLASS_NAMES[pred_idx]
    conf            = proba[pred_idx]
    correct         = pred == GROUND_TRUTH

    results.append((pred, conf, correct))
    pred_counts[pred] += 1

    elapsed   = time.time() - collection_start
    remaining = max(0, COLLECTION_SECONDS - elapsed)

    COLOURS = {'Baseline': '\033[92m', 'Quadrant_1': '\033[93m', 'Quadrant_2': '\033[91m'}
    RESET   = '\033[0m'
    colour  = COLOURS.get(pred, '')
    tick    = '✓' if correct else '✗'

    running_acc = sum(1 for _, _, ok in results if ok) / len(results) * 100

    print(
        f"  {tick} {colour}{pred:12s}{RESET} conf={conf:.2f} | "
        f"B={proba[0]:.2f} Q1={proba[1]:.2f} Q2={proba[2]:.2f} | "
        f"{remaining:.0f}s left | acc={running_acc:.1f}%"
    )


# ── Entry point ───────────────────────────────────────────────────────────────
client = mqtt.Client()
client.username_pw_set("mqtt_device", "cheese")
client.on_connect = on_connect
client.on_message = on_message

print(f"\nConnecting to 192.168.3.115:1883...")
client.connect("192.168.3.115", 1883, 60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nInterrupted.")
    client.disconnect()
