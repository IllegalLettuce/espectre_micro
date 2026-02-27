import paho.mqtt.client as mqtt
import json
import csv
from datetime import datetime
import time
import threading

CURRENT_LABEL = "movement_doors_open_q1"  # change when switching what to record

WARMUP_SECONDS = 3      # Wait time before starting collection
COLLECTION_SECONDS = 45  # How long to collect data

csv_file = open(f'csi_training_data_{CURRENT_LABEL}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', 'a', newline='')

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    'timestamp',
    'label',
    'entropy_turb',
    'iqr_turb', 
    'variance_turb',
    'skewness',
    'kurtosis',
    'amp_mean',
    'amp_range',
    'amp_std',
    'amp_mean_low',
    'amp_mean_mid',
    'amp_mean_high'
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
        
        # Start warmup timer
        threading.Timer(WARMUP_SECONDS, start_collection).start()

def start_collection():
    global collection_started, collection_start_time, warmup_complete
    warmup_complete = True
    collection_started = True
    collection_start_time = time.time()
    print(f"START COLLECTING: '{CURRENT_LABEL}' for {COLLECTION_SECONDS} seconds")
    print(f"{'='*60}\n")
    
    # Start collection timer
    threading.Timer(COLLECTION_SECONDS, stop_collection).start()

def stop_collection():
    global collection_started
    collection_started = False
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE!")
    print(f"   Label: {CURRENT_LABEL}")
    print(f"   Samples collected: {sample_count}")
    print(f"   Average rate: {sample_count/COLLECTION_SECONDS:.1f} Hz")
    print(f"{'='*60}\n")
    csv_file.close()
    client.disconnect()
    client.loop_stop()

def on_message(client, userdata, msg):
    global sample_count
    
    # Ignore messages during warmup
    if not warmup_complete:
        return
    
    # Only collect during collection window
    if not collection_started:
        return
    
    payload = json.loads(msg.payload.decode())
    
    if 'features' in payload:
        f = payload['features']
        csv_writer.writerow([
            datetime.now().isoformat(),
            CURRENT_LABEL,
            f.get('entropy_turb', 0),
            f.get('iqr_turb', 0),
            f.get('variance_turb', 0),
            f.get('skewness', 0),
            f.get('kurtosis', 0),
            f.get('amp_mean', 0),
            f.get('amp_range', 0),
            f.get('amp_std', 0),
            f.get('amp_mean_low', 0),  
            f.get('amp_mean_mid', 0), 
            f.get('amp_mean_high', 0),
        ])
        csv_file.flush()
        sample_count += 1
        
        # Show progress every 10 samples
        if sample_count % 10 == 0:
            elapsed = time.time() - collection_start_time
            remaining = COLLECTION_SECONDS - elapsed
            print(f"{sample_count} samples | {remaining:.0f}s remaining")

client = mqtt.Client()
client.username_pw_set("mqtt_device", "cheese")
client.on_connect = on_connect
client.on_message = on_message

print(f"Connecting to 192.168.3.115:1883...")
client.connect("192.168.3.115", 1883, 60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopped by user (Ctrl+C)")
    csv_file.close()
