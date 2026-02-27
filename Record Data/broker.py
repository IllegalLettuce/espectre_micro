import paho.mqtt.client as mqtt
import datetime

# ── Configuration ──────────────────────────────────────────
BROKER   = "10.172.33.143"   # e.g. your Home Assistant IP
PORT     = 1883
TOPIC    = "#"           # '#' = wildcard (all topics)
USERNAME = "mqtt_device"            # leave empty if no auth
PASSWORD = "cheese"
# ──────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc, properties=None):
    if rc.is_failure:
        print(f"[{now()}] Connection failed: {rc}")
    else:
        print(f"[{now()}] Connected successfully")
        client.subscribe(TOPIC)
        print(f"[{now()}] Subscribed to topic: '{TOPIC}'")


def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", errors="replace")
    print(f"[{now()}] [{msg.topic}] (QoS {msg.qos}) → {payload}")

def on_disconnect(client, userdata, rc, properties=None):
    print(f"[{now()}] Disconnected: {rc}")


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")

# ── Client setup ───────────────────────────────────────────
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

if USERNAME:
    client.username_pw_set(USERNAME, PASSWORD)

client.on_connect    = on_connect
client.on_message    = on_message
client.on_disconnect = on_disconnect

print(f"[{now()}] Connecting to {BROKER}:{PORT} ...")
client.connect(BROKER, PORT, keepalive=60)
client.loop_forever()
