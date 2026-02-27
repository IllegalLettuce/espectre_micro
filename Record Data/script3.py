import paho.mqtt.client as mqtt
import json
import csv
from datetime import datetime
import time
import threading

# change when switching what to record
# 0_0_0 => all doors closed

# 1_0_0 => door to living room open
# 0_1_0 => door to study room open
# 0_0_1 => door to kitchen open

# 0_1_1 => door to study room and kitchen open
# 1_1_1 => all doors open

# CURRENT_LABEL = "baseline_0_0_0" 
CURRENT_LABEL = "movement_stairs_0_0_0"
# CURRENT_LABEL = "movement_hallway_0_0_0"

WARMUP_SECONDS = 3       # Wait time before starting collection
COLLECTION_SECONDS = 45  # How long to collect data

NUM_SC_AMPS = 64
SC_AMP_COLS = [f'sc_amp_{i}' for i in range(NUM_SC_AMPS)]

csv_file = open(
    f'csi_training_data_{CURRENT_LABEL}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
    'a', newline=''
)
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    'timestamp',
    'label',
    # Turbulence features
    'entropy_turb',
    'iqr_turb',
    'variance_turb',
    # W=1 features
    'skewness',
    'kurtosis',
    # Amplitude features
    'amp_mean',
    'amp_range',
    'amp_std',
    # Subband features
    'amp_mean_low',
    'amp_mean_mid',
    'amp_mean_high',
    # Per-subcarrier amplitudes
    *SC_AMP_COLS,
    # Phase features
    'phase_mean',
    'phase_std',
    'phase_range',
])


collection_started = False
collection_start_time = None
warmup_complete = False
sample_count = 0


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT broker")
        client.subscribe("home/espectre/node1")
        print(f"Subscribed to home/espectre/node1\n")
        print(f"Warmup period: {WARMUP_SECONDS} seconds...")
        threading.Timer(WARMUP_SECONDS, start_collection).start()


def start_collection():
    global collection_started, collection_start_time, warmup_complete
    warmup_complete = True
    collection_started = True
    collection_start_time = time.time()
    
    print(f"START COLLECTING: '{CURRENT_LABEL}' for {COLLECTION_SECONDS} seconds")
    print(f"{'='*60}\n")
    threading.Timer(COLLECTION_SECONDS, stop_collection).start()


def stop_collection():
    global collection_started
    collection_started = False
    
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE!")
    print(f"   Label:           {CURRENT_LABEL}")
    print(f"   Samples:         {sample_count}")
    print(f"   Average rate:    {sample_count/COLLECTION_SECONDS:.1f} Hz")
    print(f"   Features/sample: {11 + NUM_SC_AMPS + 3} (11 base + {NUM_SC_AMPS} sc_amps + 3 phase)")
    print(f"{'='*60}\n")
    
    csv_file.close()
    client.disconnect()
    client.loop_stop()


def on_message(client, userdata, msg):
    global sample_count

    if not warmup_complete or not collection_started:
        return

    try:
        payload = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        return

    if 'features' not in payload:
        return

    f = payload['features']

    # Extract sc_amps list — pad/truncate to exactly NUM_SC_AMPS values
    sc_amps_raw = f.get('sc_amps', [])
    sc_amps = sc_amps_raw[:NUM_SC_AMPS]                      # truncate if too long
    sc_amps += [0.0] * (NUM_SC_AMPS - len(sc_amps))          # pad if too short

    csv_writer.writerow([
        datetime.now().isoformat(),
        CURRENT_LABEL,
        # Turbulence features
        f.get('entropy_turb', 0),
        f.get('iqr_turb', 0),
        f.get('variance_turb', 0),
        # W=1 features
        f.get('skewness', 0),
        f.get('kurtosis', 0),
        # Amplitude features
        f.get('amp_mean', 0),
        f.get('amp_range', 0),
        f.get('amp_std', 0),
        # Subband features
        f.get('amp_mean_low', 0),
        f.get('amp_mean_mid', 0),
        f.get('amp_mean_high', 0),
        # Per-subcarrier amplitudes (flattened)
        *sc_amps,
        # Phase features
        f.get('phase_mean', 0),
        f.get('phase_std', 0),
        f.get('phase_range', 0),
    ])
    csv_file.flush()
    sample_count += 1

    # Show progress every 10 samples — include a phase/sc sanity check
    if sample_count % 20 == 0:
        elapsed = time.time() - collection_start_time
        remaining = COLLECTION_SECONDS - elapsed
        phase_ok = f.get('phase_std', 0) != 0
        sc_ok = any(v != 0 for v in sc_amps)
        print(
            f"{sample_count} samples | {remaining:.0f}s remaining | "
            f"phase={'OK' if phase_ok else 'ZERO - check firmware'} | "
            f"sc_amps={'OK' if sc_ok else 'ZERO - check firmware'}"
        )


client = mqtt.Client()
client.username_pw_set("mqtt_device", "cheese")
client.on_connect = on_connect
client.on_message = on_message

print(f"Connecting to 192.168.3.115:1883...")
client.connect("192.168.3.115", 1883, 60)


client.loop_forever()

    
