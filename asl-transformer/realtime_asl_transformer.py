#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Realtime gesture/ASL with TFLite + MediaPipe + OpenCV


"""
Realtime gesture/ASL with TFLite + MediaPipe + OpenCV

- Read webcam
- Get keypoints from MediaPipe Holistic
- Build sequence (sliding window)
- Feed into .tflite model (input shape like (1, 32, 543, 3))
- Predict continuously, NO threshold (always predicting)
- Light smoothing to reduce jitter
"""

import cv2
import numpy as np
from collections import deque
import time
import json
import tensorflow as tf
import mediapipe as mp

# =========================
# CONFIG
# =========================

MODEL_PATH = "/home/lananh/GISLR/asl-transformer/asl_transformer_20.tflite"
LABELS_JSON_PATH = "/home/lananh/GISLR/asl-transformer/asl_transformer_20_sign_to_idx.json"

SMOOTHING_WINDOW = 3   # 3 identical frames → slightly smoother
WEBCAM_ID = 0          # 0 = default cam

# =========================
# LOAD LABELS FROM JSON
# =========================

def load_labels_from_json(path):
    """
    JSON format:
    {
        "listen": 0,
        "look": 1,
        "shhh": 2,
        ...
    }

    Returns:
        labels: list such that labels[id] = "class_name"
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON file must be a dict {class_name: id}")

    max_id = max(int(v) for v in data.values())
    labels = [""] * (max_id + 1)

    for name, idx in data.items():
        idx = int(idx)
        if 0 <= idx <= max_id:
            labels[idx] = name

    # Fill empty slots if any
    for i, v in enumerate(labels):
        if v == "":
            labels[i] = f"class_{i}"

    return labels

labels = load_labels_from_json(LABELS_JSON_PATH)
num_classes = len(labels)
print(f"[INFO] Loaded {num_classes} labels from JSON:")
print(labels)

# =========================
# LOAD TFLITE MODEL
# =========================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
output_index = output_details[0]['index']

input_shape = input_details[0]['shape']   # e.g.: [ 1 32 543 3]
print("[INFO] Raw model input shape:", input_shape)

rank = len(input_shape)

# We want: SEQ_LEN = number of frames, INPUT_FEATURES = features per frame (flattened)
if rank == 3:
    # (batch, seq_len, feature_dim)
    _, SEQ_LEN, INPUT_FEATURES = input_shape
    SEQ_LEN = int(SEQ_LEN)
    INPUT_FEATURES = int(INPUT_FEATURES)
    MODEL_INPUT_SHAPE = tuple(input_shape)

elif rank == 4:
    # (batch, seq_len, d1, d2) = (1, 32, 543, 3) for example
    _, SEQ_LEN, D1, D2 = input_shape
    SEQ_LEN = int(SEQ_LEN)
    INPUT_FEATURES = int(D1 * D2)
    MODEL_INPUT_SHAPE = tuple(input_shape)

else:
    SEQ_LEN = int(input_shape[1])
    INPUT_FEATURES = int(np.prod(input_shape[2:]))
    MODEL_INPUT_SHAPE = tuple(input_shape)

print(f"[INFO] SEQ_LEN={SEQ_LEN}, INPUT_FEATURES={INPUT_FEATURES}, MODEL_INPUT_SHAPE={MODEL_INPUT_SHAPE}")

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

# =========================
# MEDIAPIPE SETUP
# =========================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# KEYPOINTS EXTRACTION
# =========================

_warning_printed = False

def extract_keypoints(results):
    """
    Return one vector (INPUT_FEATURES ≈ 1629,) from MediaPipe Holistic.

    Standard: 543 landmarks * 3 (x,y,z) = 1629:
        - Pose: 33 * 3
        - Face: 468 * 3
        - Left hand: 21 * 3
        - Right hand: 21 * 3
    """

    # Pose: 33 * 3
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 3, dtype=np.float32)

    # Face: 468 * 3
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3, dtype=np.float32)

    # Left hand: 21 * 3
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    # Right hand: 21 * 3
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    keypoints = np.concatenate([pose, face, lh, rh]).astype(np.float32)

    global _warning_printed
    if keypoints.shape[0] != INPUT_FEATURES and not _warning_printed:
        print(f"[WARN] keypoints dim = {keypoints.shape[0]}, INPUT_FEATURES = {INPUT_FEATURES}")
        _warning_printed = True

    return keypoints  # (INPUT_FEATURES,)

# =========================
# INFERENCE FUNCTION
# =========================

def predict_sequence(seq_window):
    """
    seq_window: deque containing SEQ_LEN vectors (INPUT_FEATURES,)
    return: (pred_label_id, pred_confidence, probs_vector)
    """
    arr = np.array(seq_window, dtype=np.float32)     # (SEQ_LEN, INPUT_FEATURES)
    seq = arr.reshape(MODEL_INPUT_SHAPE)             # (1, SEQ_LEN, 543, 3) or similar

    interpreter.set_tensor(input_index, seq)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_index)[0]  # (num_classes,)

    probs = softmax(raw_output)
    pred_id = int(np.argmax(probs))
    conf = float(probs[pred_id])

    # debug
    print(f"[DEBUG] pred = {labels[pred_id]}, conf = {conf:.3f}")

    return pred_id, conf, probs

# =========================
# REALTIME LOOP
# =========================

def main():
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    seq_window = deque(maxlen=SEQ_LEN)
    smooth_preds = deque(maxlen=SMOOTHING_WINDOW)

    current_label = ""
    current_conf = 0.0

    print("[INFO] Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Cannot read frame from webcam")
                break

            # Flip horizontally like a mirror
            frame = cv2.flip(frame, 1)

            # BGR -> RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True

            # Draw landmarks (optional)
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

            # Extract keypoints
            keypoints = extract_keypoints(results)
            seq_window.append(keypoints)

            # Only predict when we have SEQ_LEN frames
            if len(seq_window) == SEQ_LEN:
                pred_id, conf, probs = predict_sequence(seq_window)

                # always use pred_id, no threshold
                smooth_preds.append(pred_id)

                if len(smooth_preds) == SMOOTHING_WINDOW:
                    values, counts = np.unique(list(smooth_preds), return_counts=True)
                    majority_id = int(values[np.argmax(counts)])
                    current_label = labels[majority_id]
                    current_conf = conf  # show confidence of the latest step

            # =========================
            # DRAW UI
            # =========================
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

            text = f"Label: {current_label} | Conf: {current_conf:.2f}"
            cv2.putText(
                frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
            )

            cv2.putText(
                frame, f"len={len(seq_window)}/{SEQ_LEN}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA
            )

            cv2.imshow("Realtime Transformer (TFLite)", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()
        print("[INFO] Program exited")

if __name__ == "__main__":
    main()
