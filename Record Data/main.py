import paho.mqtt.client as mqtt
import json
import csv
from datetime import datetime

CURRENT_LABEL = "baseline" # change when switching what to record

csv_file = open('csi_training_data.csv', 'a', newline='')
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

def on_message(client, userdata, msg):
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
        print(f"Logged sample: {CURRENT_LABEL}")

client = mqtt.Client()
client.username_pw_set("mqtt_device", "cheese")
client.on_message = on_message
client.connect("192.168.3.115", 1883, 60)
client.subscribe("home/espectre/node1")

print(f"Collecting data for label: {CURRENT_LABEL}")

client.loop_forever()
