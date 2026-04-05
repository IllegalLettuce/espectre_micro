import appdaemon.plugins.hass.hassapi as hass
import json
import os
import traceback
import datetime
import numpy as np
from collections import deque
import base64
import struct


VALID_SC_INDICES = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                    21,22,23,24,25,26,34,35,36,37,38,39,40,41,42,43]

AGG_FEATURES = [
    "entropy_turb", "iqr_turb", "variance_turb",       # indices 0, 1, 2
    "skewness", "kurtosis",                              # indices 3, 4
    "amp_mean", "amp_range", "amp_std",                  # indices 5, 6, 7
    "amp_mean_low", "amp_mean_mid", "amp_mean_high",     # indices 8, 9, 10
    "phase_diff_mean", "phase_diff_std",                 # indices 11, 12
    "phase_diff_range", "phase_diff_skew",               # indices 13, 14
]

# Must match train7.py exactly
LINEAR_NORM_INDICES    = {0, 1, 5, 6, 7, 8, 9, 10}   # ÷ sc_global_mean
QUADRATIC_NORM_INDICES = {2}                            # ÷ sc_global_mean²
# Indices 3, 4, 11-14 are dimensionless — kept raw


class CSIClassifier(hass.Hass):

    def initialize(self):
        self.log("===== CSI Classifier Starting =====")

        try:
            import joblib
            # ✓ No scaler — RF is scale-invariant.
            #   All normalisation is done per-window in _extract_window_features.
            self.model = joblib.load("/config/apps/rf_spatial_classifier.pkl")

            # Load label map from pkl — robust to future class additions
            raw_map          = joblib.load("/config/apps/rf_label_mapping.pkl")
            self.class_names = {v: k for k, v in raw_map.items()}   # idx → label

            self.log(f"Model loaded successfully — classes: {self.class_names}")
        except Exception as e:
            self.log(f"ERROR: Failed to load model: {e}", level="ERROR")
            return

        self.SEQ_LEN        = 28
        self.STRIDE         = 8
        self.stride_counter = 0

        self.sc_buffer  = deque(maxlen=self.SEQ_LEN)
        self.agg_buffer = deque(maxlen=self.SEQ_LEN)

        self.last_frame_time      = None
        self.last_published_state = None
        self.publish_counter      = 0

        self.listen_event(
            self.on_mqtt_message, "MQTT_MESSAGE", namespace="mqtt_ns"
        )
        self.log("CSI Classifier ready — 28-frame windowed RF, stride=8, 142 features")

    def _extract_window_features(self, sc_win, agg_win):
        """
        Must exactly mirror build_windowed_features() in train7.py.

        Per-window normalisation:
          - Amplitude-linear aggregates (entropy, iqr, amp family): ÷ sc_global_mean
          - variance_turb:                                          ÷ sc_global_mean²
          - skewness, kurtosis, phase_diff_*:                       raw (dimensionless)
          - Per-subcarrier amplitudes:                              ÷ sc_global_mean
        """
        sc_global_mean  = float(np.mean(sc_win)) + 1e-6
        sc_global_mean2 = sc_global_mean ** 2

        feats = []

        # 1. Aggregate feature stats — 15 × 5 = 75
        for i in range(agg_win.shape[1]):
            col = agg_win[:, i].copy()
            if i in QUADRATIC_NORM_INDICES:
                col = col / sc_global_mean2
            elif i in LINEAR_NORM_INDICES:
                col = col / sc_global_mean
            # else: raw (skewness, kurtosis, phase_diff_*)
            feats += [
                float(np.mean(col)),
                float(np.std(col)),
                float(np.max(col)),
                float(np.percentile(col, 25)),
                float(np.percentile(col, 75)),
            ]

        # 2. Per-subcarrier normalised mean + std — 31 × 2 = 62
        sc_norm = sc_win / sc_global_mean
        feats.extend(np.mean(sc_norm, axis=0).tolist())
        feats.extend(np.std(sc_norm,  axis=0).tolist())

        # 3. Spike features on variance_turb (already ÷ sc_global_mean²) — 4
        vt      = agg_win[:, 2] / sc_global_mean2
        vt_mean = float(np.mean(vt))
        vt_max  = float(np.max(vt))
        vt_std  = float(np.std(vt))
        feats += [
            vt_max,
            vt_std,
            float(np.sum(vt > vt_mean + 2 * vt_std)),
            float(vt_max / (vt_mean + 1e-9)),
        ]

        # 4. Temporal trend — 1
        half = self.SEQ_LEN // 2
        feats.append(float(np.mean(vt[half:]) - np.mean(vt[:half])))

        # Total: 75 + 62 + 4 + 1 = 142
        return np.array(feats, dtype='float32')

    def on_mqtt_message(self, event_name, data, kwargs):
        try:
            if data.get("topic") != "home/espectre/node1":
                return

            msg          = json.loads(data.get("payload", "{}"))
            features_src = msg.get("features", {})
            if not features_src:
                return

            # ── Flush buffer on gap > 500ms ───────────────────────────
            now = datetime.datetime.now()
            if self.last_frame_time is not None:
                if (now - self.last_frame_time).total_seconds() > 0.5:
                    self.sc_buffer.clear()
                    self.agg_buffer.clear()
                    self.stride_counter = 0
                    self.log("Buffer flushed — frame gap detected")
            self.last_frame_time = now
            # ─────────────────────────────────────────────────────────

            # ── Decode subcarrier amplitudes ──────────────────────────
            sc_amps_raw = features_src.get("sc_amps", "")
            if isinstance(sc_amps_raw, str) and sc_amps_raw:
                buf     = base64.b64decode(sc_amps_raw)
                sc_amps = [struct.unpack_from('>H', buf, i * 2)[0] / 100.0
                           for i in range(len(buf) // 2)]
            else:
                sc_amps = sc_amps_raw if isinstance(sc_amps_raw, list) else []

            sc_frame = np.array(
                [float(sc_amps[i]) if i < len(sc_amps) else 0.0
                 for i in VALID_SC_INDICES], dtype='float32')

            agg_frame = np.array(
                [float(features_src.get(f, 0.0)) for f in AGG_FEATURES],
                dtype='float32')

            self.sc_buffer.append(sc_frame)
            self.agg_buffer.append(agg_frame)

            if len(self.sc_buffer) < self.SEQ_LEN:
                return

            # ── Stride gate ───────────────────────────────────────────
            self.stride_counter += 1
            if self.stride_counter % self.STRIDE != 0:
                return
            # ─────────────────────────────────────────────────────────

            feat = self._extract_window_features(
                np.array(self.sc_buffer),
                np.array(self.agg_buffer)
            ).reshape(1, -1)

            # ✓ Single forward pass — no double inference
            probabilities = self.model.predict_proba(feat)[0]
            prediction    = int(np.argmax(probabilities))
            confidence    = float(probabilities[prediction])
            location      = self.class_names.get(prediction, f"Unknown_{prediction}")

            self.log(
                f"PRED: {location} | "
                + "  ".join(
                    f"{self.class_names[i]}={probabilities[i]:.2f}"
                    for i in sorted(self.class_names)
                )
                + f"  | conf={confidence:.2f}"
            )

            # ── Publish on change OR every 3 predictions (~1s cadence) ──
            self.publish_counter += 1
            if location != self.last_published_state or self.publish_counter >= 3:
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
                            self.class_names[i]: f"{probabilities[i]*100:.1f}%"
                            for i in range(len(probabilities))
                        },
                    },
                )
                self.last_published_state = location
                self.publish_counter = 0

        except Exception as e:
            self.log(f"ERROR in on_mqtt_message: {e}", level="ERROR")
            self.log(traceback.format_exc(), level="ERROR")