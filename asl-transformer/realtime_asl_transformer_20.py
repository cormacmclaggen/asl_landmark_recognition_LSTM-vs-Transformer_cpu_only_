#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Realtime ASL recognition with TFLite + MediaPipe + OpenCV

import os
import json
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf

# ==============================
# CONFIG
# ==============================

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

ROWS_PER_FRAME = 543
N_FRAMES_WINDOW = 32   # phải trùng với MAX_FRAMES lúc train

# ==== SỬA 3 DÒNG NÀY NẾU PATH KHÁC ====
MODEL_PATH = "/home/lananh/GISLR/asl-transformer/asl_transformer_20.tflite"
SIGN_MAP_PATH = "/home/lananh/GISLR/asl-transformer/asl_transformer_20_sign_to_idx.json"
TEMPLATE_PQ = "/home/lananh/GISLR/train_landmark_files/16069/695046.parquet"
# Nếu file này không tồn tại, đổi sang 1 parquet có thật khác trong train_landmark_files.

MIN_CONF_FOR_UPDATE = 0.0   # cho debug, muốn ổn định hơn thì tăng lên 0.3


# ==============================
# 1. Skeleton từ parquet template
# ==============================

def get_xyz_skeleton(template_parquet_path: str) -> pd.DataFrame:
    xyz = pd.read_parquet(template_parquet_path)
    xyz_skel = xyz[["type", "landmark_index"]].drop_duplicates().reset_index(drop=True)
    return xyz_skel


# ==============================
# 2. Dataframe 1 frame
# ==============================

def create_frame_landmark_df(results, frame_idx: int, xyz_skel: pd.DataFrame) -> pd.DataFrame:
    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()

    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]

    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]

    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]

    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]

    face = face.reset_index().rename(columns={"index": "landmark_index"}).assign(type="face")
    pose = pose.reset_index().rename(columns={"index": "landmark_index"}).assign(type="pose")
    left_hand = left_hand.reset_index().rename(columns={"index": "landmark_index"}).assign(type="left_hand")
    right_hand = right_hand.reset_index().rename(columns={"index": "landmark_index"}).assign(type="right_hand")

    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
    landmarks = xyz_skel.merge(landmarks, on=["type", "landmark_index"], how="left")
    landmarks = landmarks.assign(frame=frame_idx)

    return landmarks


# ==============================
# 3. TFLite + label map
# ==============================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_index = input_details["index"]
output_index = output_details["index"]

input_shape = input_details["shape"]
input_rank = len(input_shape)
input_dtype = input_details["dtype"]

print("[INFO] Model input shape:", input_shape)
print("[INFO] Input rank:", input_rank, "dtype:", input_dtype)

with open(SIGN_MAP_PATH, "r") as f:
    SIGN2IDX = json.load(f)

# JSON dạng {"listen": 0, "look": 1, ...}
IDX2SIGN = {int(v): k for k, v in SIGN2IDX.items()}
print("[INFO] Loaded", len(IDX2SIGN), "labels:", [IDX2SIGN[i] for i in sorted(IDX2SIGN.keys())])


# ==============================
# 4. Predict từ buffer
# ==============================

def predict_from_buffer(frames_dfs):
    """
    Return: best_label, best_prob, top3_list
    """
    if len(frames_dfs) < N_FRAMES_WINDOW:
        return "...", 0.0, []

    recent_dfs = frames_dfs[-N_FRAMES_WINDOW:]

    fixed_dfs = []
    for i, df in enumerate(recent_dfs):
        tmp = df.copy()
        tmp["frame"] = i
        fixed_dfs.append(tmp)

    df_all = pd.concat(fixed_dfs).reset_index(drop=True)
    df_all = df_all.sort_values(["frame", "type", "landmark_index"]).reset_index(drop=True)

    xyz = df_all[["x", "y", "z"]].to_numpy()
    n_frames = df_all["frame"].nunique()

    expected_len = n_frames * ROWS_PER_FRAME
    if xyz.shape[0] != expected_len:
        print(f"[WARN] xyz len {xyz.shape[0]} != {expected_len}")
        return "?", 0.0, []

    xyz = xyz.reshape(n_frames, ROWS_PER_FRAME, 3).astype(np.float32)

    # Chuẩn hoá giống lúc train
    mean = np.nanmean(xyz, axis=(0, 1), keepdims=True)
    std = np.nanstd(xyz, axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1e-6
    xyz = (xyz - mean) / std
    xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)

    # clamp cho chắc
    xyz = np.clip(xyz, -5.0, 5.0)

    try:
        if input_rank == 3:
            model_input = xyz.astype(input_dtype)               # (32, 543, 3)
        else:
            model_input = np.expand_dims(xyz, axis=0).astype(input_dtype)  # (1, 32, 543, 3)

        interpreter.set_tensor(input_index, model_input)
        interpreter.invoke()

        outputs = interpreter.get_tensor(output_index)[0].astype(np.float32)   # (20,)

        if not np.all(np.isfinite(outputs)):
            print("[WARN] model outputs NaN/Inf. min=", np.nanmin(outputs), "max=", np.nanmax(outputs))
            return "...", 0.0, []

        # Probabilities – the model likely already applied softmax,
        # but we apply it again to be safe
        exp = np.exp(outputs - np.max(outputs))
        probs = exp / (np.sum(exp) + 1e-8)

        if not np.all(np.isfinite(probs)):
            print("[WARN] probs NaN/Inf.")
            return "...", 0.0, []

        idx_sorted = np.argsort(probs)[::-1]
        top3_idx = idx_sorted[:3]
        top3 = []
        for i in top3_idx:
            label = IDX2SIGN.get(int(i), f"IDX_{int(i)}")
            top3.append((label, float(probs[i])))

        best_label, best_prob = top3[0]

        debug_str = " | ".join([f"{lbl}:{p:.2f}" for lbl, p in top3])
        print(f"[DEBUG] Top3 => {debug_str}")

        return best_label, best_prob, top3

    except Exception as e:
        print("[ERROR] Prediction failed:", e)
        return "?", 0.0, []


# ==============================
# 5. Main loop
# ==============================

def main():
    if not os.path.exists(TEMPLATE_PQ):
        raise FileNotFoundError(f"Template parquet not found: {TEMPLATE_PQ}")

    xyz_skel = get_xyz_skeleton(TEMPLATE_PQ)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam at index 0. Try 1 or 2.")
        return

    frames_dfs = []
    current_sign = "..."
    current_conf = 0.0
    current_top3 = []

    frame_counter = 0

    prev_time = time.time()
    fps = 0.0

    print("✅ Webcam started. ESC để thoát.")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 1.0 / dt

            frame_counter += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            image.flags.writeable = True
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                image_bgr,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            frame_df = create_frame_landmark_df(results, frame_counter, xyz_skel)
            frames_dfs.append(frame_df)
            if len(frames_dfs) > N_FRAMES_WINDOW:
                frames_dfs.pop(0)

            best_label, best_prob, top3 = predict_from_buffer(frames_dfs)

            if best_label not in ["?", "..."] and best_prob >= MIN_CONF_FOR_UPDATE:
                current_sign = best_label
                current_conf = best_prob
                current_top3 = top3

            display = cv2.flip(image_bgr, 1)

            h, w, _ = display.shape
            cv2.rectangle(display, (0, 0), (w, 160), (0, 0, 0), -1)

            text_main = f"Predicted: {current_sign}  (conf: {current_conf:.2f})"
            cv2.putText(
                display, text_main, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA,
            )

            text_info = f"FPS: {fps:5.1f}   Buf: {len(frames_dfs)}/{N_FRAMES_WINDOW}"
            cv2.putText(
                display, text_info, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
            )

            y0 = 105
            for i, (lbl, p) in enumerate(current_top3):
                line = f"{i+1}. {lbl}: {p:.2f}"
                cv2.putText(
                    display, line, (20, y0 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

            cv2.putText(
                display, "ESC to quit",
                (w - 200, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow("Realtime ASL Transformer (20 classes)", display)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Bye!")


if __name__ == "__main__":
    main()
