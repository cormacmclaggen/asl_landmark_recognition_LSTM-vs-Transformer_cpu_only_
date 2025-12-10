#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf

# ==============================
# CONFIG – SỬA CHO ĐÚNG
# ==============================

# Video cần nhận dạng
VIDEO_PATH = "/home/lananh/GISLR/test_video.mp4"

# Model & label map
MODEL_PATH = "/home/lananh/GISLR/model.tflite"
SIGN_MAP_PATH = "/home/lananh/GISLR/sign_to_prediction_index_map.json"

# Một parquet bất kỳ trong train_landmark_files để lấy skeleton (type + landmark_index)
TEMPLATE_PQ = "/home/lananh/GISLR/train_landmark_files/16069/695046.parquet"

ROWS_PER_FRAME = 543
MAX_FRAMES = 32  # phải trùng với lúc train (32 frame / sample)


# ==============================
# MediaPipe setup
# ==============================

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


# ==============================
# 1. Đọc skeleton từ parquet
# ==============================

def get_xyz_skeleton(template_parquet_path: str) -> pd.DataFrame:
    xyz = pd.read_parquet(template_parquet_path)
    xyz_skel = xyz[["type", "landmark_index"]].drop_duplicates().reset_index(drop=True)
    return xyz_skel


# ==============================
# 2. Landmarks cho 1 frame
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
# 3. Load TFLite + labels
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

with open(SIGN_MAP_PATH, "r") as f:
    SIGN2IDX = json.load(f)

IDX2SIGN = {int(v): k for k, v in SIGN2IDX.items()}
print("[INFO] Loaded", len(IDX2SIGN), "labels from JSON:")
print("       ", [IDX2SIGN[i] for i in sorted(IDX2SIGN.keys())])


# ==============================
# 4. Video → (MAX_FRAMES, 543, 3)
# ==============================

def video_to_sequence_xyz(video_path: str, xyz_skel: pd.DataFrame) -> np.ndarray:
    """
    Đọc toàn bộ VIDEO_PATH, trích landmark cho từng frame,
    rồi sample/pad về (MAX_FRAMES, 543, 3), chuẩn hoá như lúc train.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames_dfs = []
    frame_idx = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # BGR -> RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            frame_df = create_frame_landmark_df(results, frame_idx, xyz_skel)
            frames_dfs.append(frame_df)

    cap.release()

    if len(frames_dfs) == 0:
        raise RuntimeError("No frames were processed from the video.")

    print(f"[INFO] Total frames with landmarks: {len(frames_dfs)}")

    # ---- Sample / pad về MAX_FRAMES ----
    if len(frames_dfs) >= MAX_FRAMES:
        # Lấy đều MAX_FRAMES frame trong cả video
        idxs = np.linspace(0, len(frames_dfs) - 1, MAX_FRAMES).astype(int)
        selected = [frames_dfs[i] for i in idxs]
    else:
        # Nếu video ngắn, lặp lại frame cuối để đủ MAX_FRAMES
        selected = frames_dfs.copy()
        while len(selected) < MAX_FRAMES:
            selected.append(selected[-1])

    # Dồn lại giống predict_from_buffer trước đây
    fixed_dfs = []
    for i, df in enumerate(selected):
        tmp = df.copy()
        tmp["frame"] = i   # 0..MAX_FRAMES-1
        fixed_dfs.append(tmp)

    df_all = pd.concat(fixed_dfs).reset_index(drop=True)
    df_all = df_all.sort_values(["frame", "type", "landmark_index"]).reset_index(drop=True)

    xyz = df_all[["x", "y", "z"]].to_numpy()
    n_frames = df_all["frame"].nunique()

    expected_len = n_frames * ROWS_PER_FRAME
    if xyz.shape[0] != expected_len:
        raise RuntimeError(f"xyz len {xyz.shape[0]} != {expected_len}")

    xyz = xyz.reshape(n_frames, ROWS_PER_FRAME, 3).astype(np.float32)

    # Chuẩn hoá giống lúc train
    mean = np.nanmean(xyz, axis=(0, 1), keepdims=True)
    std = np.nanstd(xyz, axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1e-6
    xyz = (xyz - mean) / std
    xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)

    # Clamp nhẹ cho an toàn
    xyz = np.clip(xyz, -5.0, 5.0)

    return xyz  # (MAX_FRAMES, 543, 3)


# ==============================
# 5. Inference cho 1 sequence
# ==============================

def infer_sequence_xyz(xyz: np.ndarray):
    """
    xyz: (MAX_FRAMES, 543, 3)
    Trả về best_label, best_prob, top3
    """
    if input_rank == 3:
        # (MAX_FRAMES, 543, 3) -> y như input Keras nếu nó là (32, 543, 3)
        model_input = xyz.astype(input_dtype)
    else:
        # (1, MAX_FRAMES, 543, 3)
        model_input = np.expand_dims(xyz, axis=0).astype(input_dtype)

    interpreter.set_tensor(input_index, model_input)
    interpreter.invoke()

    outputs = interpreter.get_tensor(output_index)[0].astype(np.float32)

    if not np.all(np.isfinite(outputs)):
        print("[WARN] model outputs contain NaN/Inf. min=", np.nanmin(outputs), "max=", np.nanmax(outputs))
        return "?", 0.0, []

    # Softmax
    exp = np.exp(outputs - np.max(outputs))
    probs = exp / (np.sum(exp) + 1e-8)

    if not np.all(np.isfinite(probs)):
        print("[WARN] probs contain NaN/Inf. min=", np.nanmin(probs), "max=", np.nanmax(probs))
        return "?", 0.0, []

    idx_sorted = np.argsort(probs)[::-1]
    top3_idx = idx_sorted[:3]

    top3 = []
    for i in top3_idx:
        label = IDX2SIGN.get(int(i), f"IDX_{int(i)}")
        top3.append((label, float(probs[i])))

    best_label, best_prob = top3[0]
    return best_label, best_prob, top3


# ==============================
# 6. Main
# ==============================

def main():
    if not os.path.exists(TEMPLATE_PQ):
        raise FileNotFoundError(f"Template parquet not found: {TEMPLATE_PQ}")

    xyz_skel = get_xyz_skeleton(TEMPLATE_PQ)

    print("[INFO] Extracting sequence from video:", VIDEO_PATH)
    xyz_seq = video_to_sequence_xyz(VIDEO_PATH, xyz_skel)  # (MAX_FRAMES, 543, 3)
    print("[INFO] xyz_seq shape:", xyz_seq.shape)

    best_label, best_prob, top3 = infer_sequence_xyz(xyz_seq)

    print("\n===== RESULT =====")
    print(f"Best: {best_label} (conf = {best_prob:.4f})")
    print("Top-3:")
    for lbl, p in top3:
        print(f"  - {lbl}: {p:.4f}")


if __name__ == "__main__":
    main()
