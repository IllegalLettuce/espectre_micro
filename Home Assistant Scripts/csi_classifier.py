import appdaemon.plugins.hass.hassapi as hass
import json, os, traceback
import numpy as np
from collections import deque

VALID_SC_INDICES = list(range(6, 27)) + list(range(36, 59))  # 44 subcarrier

AGG_FEATURES = [
    "entropy_turb", "iqr_turb", "variance_turb",
    "skewness", "kurtosis",
    "amp_mean", "amp_range", "amp_std",
    "amp_mean_low", "amp_mean_mid", "amp_mean_high",
    'phase_mean', 'phase_std', 'phase_range',
]

class CSIClassifier(hass.Hass):

    def initialize(self):
        self.log("===== CSI Classifier Starting =====")

        try:
            import joblib
            self.model  = joblib.load("/config/apps/rf_spatial_classifier.pkl")
            self.scaler = joblib.load("/config/apps/rf_scaler.pkl")
            self.log("Model + scaler loaded successfully")
        except Exception as e:
            self.log(f"ERROR: Failed to load model/scaler: {e}", level="ERROR")
            return

        # ── NEW: sliding window buffers ──────────────────────────────
        self.SEQ_LEN    = 28           # ~1s at 28.6 Hz
        self.sc_buffer  = deque(maxlen=self.SEQ_LEN)
        self.agg_buffer = deque(maxlen=self.SEQ_LEN)
        self.last_frame_time      = None
        self.last_published_state = None
        self.publish_counter      = 0
        # ─────────────────────────────────────────────────────────────

        self.class_names = {0: "Baseline", 1: "Quadrant_1", 2: "Quadrant_2"}

        self.listen_event(
            self.on_mqtt_message, "MQTT_MESSAGE", namespace="mqtt_ns"
        )
        self.log("CSI Classifier ready — 28-frame windowed RF (148 features)")

    def _extract_window_features(self, sc_win, agg_win):
        """Identical logic to build_windowed_features() in test4.py."""
        feats = []
        for i in range(agg_win.shape[1]):
            col = agg_win[:, i]
            feats += [float(np.mean(col)), float(np.std(col)),
                      float(np.max(col)),
                      float(np.percentile(col, 25)),
                      float(np.percentile(col, 75))]
        feats.extend(np.mean(sc_win, axis=0).tolist())
        feats.extend(np.std(sc_win,  axis=0).tolist())
        vt      = agg_win[:, 2]
        vt_mean = float(np.mean(vt))
        vt_max  = float(np.max(vt))
        vt_std  = float(np.std(vt))
        feats += [vt_max, vt_std,
                  float(np.sum(vt > vt_mean + 2*vt_std)),
                  float(vt_max / (vt_mean + 1e-6))]
        half = self.SEQ_LEN // 2
        feats.append(float(np.mean(vt[half:]) - np.mean(vt[:half])))
        return np.array(feats, dtype='float32')

    def on_mqtt_message(self, event_name, data, kwargs):
        try:
            if data.get("topic") != "home/espectre/node1":
                return
            msg          = json.loads(data.get("payload", "{}"))
            features_src = msg.get("features", {})
            if not features_src:
                return

            # ── Flush buffer on gap > 500ms ──────────────────────────
            import datetime
            now = datetime.datetime.now()
            if self.last_frame_time is not None:
                if (now - self.last_frame_time).total_seconds() > 0.5:
                    self.sc_buffer.clear()
                    self.agg_buffer.clear()
                    self.log("Buffer flushed — frame gap detected")
            self.last_frame_time = now
            # ─────────────────────────────────────────────────────────

            sc_amps = features_src.get("sc_amps", [])
            sc_frame = np.array(
                [float(sc_amps[i]) if i < len(sc_amps) else 0.0
                 for i in VALID_SC_INDICES], dtype='float32')
            agg_frame = np.array(
                [float(features_src.get(f, 0.0)) for f in AGG_FEATURES],
                dtype='float32')

            self.sc_buffer.append(sc_frame)
            self.agg_buffer.append(agg_frame)

            if len(self.sc_buffer) < self.SEQ_LEN:
                return   # still filling — silent, no log spam

            # ── Build window feature vector ───────────────────────────
            feat    = self._extract_window_features(
                        np.array(self.sc_buffer),
                        np.array(self.agg_buffer)).reshape(1, -1)
            feat_sc = self.scaler.transform(feat)
            # ─────────────────────────────────────────────────────────

            prediction    = int(self.model.predict(feat_sc)[0])
            probabilities = self.model.predict_proba(feat_sc)[0]
            confidence    = float(probabilities[prediction])
            location      = self.class_names.get(prediction, f"Unknown_{prediction}")

            self.log(
                f"PRED: {location} | "
                f"Baseline={probabilities[0]:.2f} "
                f"Q1={probabilities[1]:.2f} "
                f"Q2={probabilities[2]:.2f} "
                f"| conf={confidence:.2f}"
            )

            # ── Throttle: publish on change OR every 10 frames ────────
            self.publish_counter += 1
            if location != self.last_published_state or self.publish_counter >= 10:
                self.set_state(
                    "sensor.csi_location",
                    state=location,
                    attributes={
                        "friendly_name":  "CSI Location",
                        "confidence":     f"{confidence*100:.1f}%",
                        "confidence_raw": confidence,
                        "class_id":       prediction,
                        "raw_class":      location,
                        "icon":           "mdi:map-marker-radius",
                        "all_probabilities": {
                            "Baseline":   f"{probabilities[0]*100:.1f}%",
                            "Quadrant_1": f"{probabilities[1]*100:.1f}%",
                            "Quadrant_2": f"{probabilities[2]*100:.1f}%",
                        },
                    },
                )
                self.last_published_state = location
                self.publish_counter = 0

        except Exception as e:
            self.log(f"ERROR in on_mqtt_message: {e}", level="ERROR")
            self.log(traceback.format_exc(), level="ERROR")
