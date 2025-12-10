#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Realtime ASL Transformer (aligned với script train của m)

- Input TFLite: (1, 32, 543, 3)
- Mỗi frame: 543 landmark (pose 33, face 468, left hand 21, right hand 21)
- Gom 32 frame → (32, 543, 3)
- Normalize per-sample: (x - mean) / std giống hệt load_parquet()
- Dự đoán liên tục, có smoothing nhẹ
"""

import cv2
import numpy as np
from collections import deque
import json
import tensorflow as tf
import mediapipe as mp

# =========================
# CONFIG
# =========================

MODEL_PATH = "/home/lananh/GISLR/asl-transformer/asl_transformer_20.tflite"
LABELS_JSON_PATH = "/home/lananh/GISLR/asl-transformer/asl_transformer_20_sign_to_idx.json"

ROWS_PER_FRAME = 543   # phải khớp với train
MAX_FRAMES = 32        # = SEQ_LEN
WEBCAM_ID = 0

SMOOTHING_WINDOW = 5   # frame giống nhau để mượt
THRESHOLD = 0.3        # conf > 0.3 mới nhận (sau khi align rồi)

# =========================
# LOAD LABELS
# =========================

def load_labels_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON phải là dict {sign_name: index}")

    max_id = max(int(v) for v in data.values())
    labels = [""] * (max_id + 1)
    for name, idx in data.items():
        idx = int(idx)
        if 0 <= idx <= max_id:
            labels[idx] = name

    for i, v in enumerate(labels):
        if v == "":
            labels[i] = f"class_{i}"

    return labels

labels = load_labels_from_json(LABELS_JSON_PATH)
print("[INFO] Loaded labels:", labels)

# =========================
# LOAD TFLITE MODEL
# =========================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

input_shape = input_details[0]["shape"]   # expect (1, 32, 543, 3)
print("[INFO] TFLite input shape:", input_shape)

# Just to be safe:
_, T, L, C = input_shape
T = int(T)
L = int(L)
C = int(C)

assert T == MAX_FRAMES, f"MAX_FRAMES mismatch: {T} vs {MAX_FRAMES}"
assert L == ROWS_PER_FRAME, f"ROWS_PER_FRAME mismatch: {L} vs {ROWS_PER_FRAME}"
assert C == 3, "Expect 3 channels (x,y,z)"

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
# 1 FRAME → (543, 3)
# =========================

def extract_frame_landmarks(results):
    """
    Build (543, 3) theo thứ tự:
    [pose(33), face(468), left(21), right(21)]
    Giống GISLR & MediaPipe Holistic.
    """

    # Pose: 33
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.pose_landmarks.landmark], dtype=np.float32)
    else:
        pose = np.zeros((33, 3), dtype=np.float32)

    # Face: 468
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.face_landmarks.landmark], dtype=np.float32)
    else:
        face = np.zeros((468, 3), dtype=np.float32)

    # Left hand: 21
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
    else:
        lh = np.zeros((21, 3), dtype=np.float32)

    # Right hand: 21
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
    else:
        rh = np.zeros((21, 3), dtype=np.float32)

    # Ghép lại
    frame = np.concatenate([pose, face, lh, rh], axis=0)  # (543, 3)

    if frame.shape != (ROWS_PER_FRAME, 3):
        print("[WARN] frame shape mismatch:", frame.shape)

    return frame

# =========================
# PREPROCESS SEQUENCE (MATCH load_parquet)
# =========================

def preprocess_sequence(seq_32x543x3):
    """
    seq_32x543x3: (32, 543, 3)
    Apply y chang code train:
        mean = np.nanmean(arr, axis=(0,1))
        std = np.nanstd(arr, axis=(0,1))
        arr = (arr - mean) / std
        nan_to_num
    """
    arr = seq_32x543x3.astype(np.float32)

    mean = np.nanmean(arr, axis=(0, 1), keepdims=True)
    std = np.nanstd(arr, axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1e-6

    arr = (arr - mean) / std
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr

# =========================
# INFERENCE
# =========================

def predict_sequence(seq_window):
    """
    seq_window: deque length 32, mỗi phần tử shape (543, 3)
    """
    # (32, 543, 3)
    seq = np.stack(seq_window, axis=0)   # (T, L, C)
    seq = preprocess_sequence(seq)       # match train

    # Reshape về (1, 32, 543, 3)
    seq = seq.reshape(input_shape).astype(np.float32)

    interpreter.set_tensor(input_index, seq)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_index)[0]  # (num_classes,)

    probs = softmax(logits)
    pred_id = int(np.argmax(probs))
    conf = float(probs[pred_id])

    return pred_id, conf, probs

# =========================
# MAIN LOOP
# =========================

def main():
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    seq_window = deque(maxlen=MAX_FRAMES)
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

            frame = cv2.flip(frame, 1)

            # Run MediaPipe
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
            # Face không nhất thiết phải vẽ cho nhẹ

            # Lấy (543, 3) cho frame hiện tại
            landmarks_543x3 = extract_frame_landmarks(results)
            seq_window.append(landmarks_543x3)

            # Đủ 32 frame mới predict
            if len(seq_window) == MAX_FRAMES:
                pred_id, conf, probs = predict_sequence(seq_window)
                # debug nhẹ:
                # print(f"[DEBUG] pred={labels[pred_id]}, conf={conf:.3f}")

                if conf > THRESHOLD:
                    smooth_preds.append(pred_id)
                    if len(smooth_preds) == SMOOTHING_WINDOW:
                        values, counts = np.unique(list(smooth_preds), return_counts=True)
                        majority_id = int(values[np.argmax(counts)])
                        current_label = labels[majority_id]
                        current_conf = conf
                else:
                    smooth_preds.clear()

            # UI overlay
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

            text = f"Label: {current_label} | Conf: {current_conf:.2f}"
            cv2.putText(
                frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
            )

            cv2.putText(
                frame, f"len={len(seq_window)}/{MAX_FRAMES}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA
            )

            cv2.imshow("Realtime ASL Transformer", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()
        print("[INFO] Exit")

if __name__ == "__main__":
    main()
