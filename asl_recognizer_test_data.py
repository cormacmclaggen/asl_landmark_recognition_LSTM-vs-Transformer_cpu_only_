#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ASL recognizer test data loader
import os
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf

# ============================
# Config
# ============================

ROWS_PER_FRAME = 543
MAX_FRAMES = 32  # giống lúc train

MODEL_PATH = "model.tflite"
SIGN_MAP_PATH = "sign_to_prediction_index_map.json"


# ============================
# Load parquet -> numpy
# ============================

def load_relevant_data_subset(pq_path: str) -> np.ndarray:
    """
    Đọc file parquet của ASL (x, y, z) và reshape về dạng
    (num_frames, ROWS_PER_FRAME, 3)
    """
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def pad_or_trim(xyz: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """
    Pad hoặc cắt số frame về max_frames.
    """
    n_frames = xyz.shape[0]
    if n_frames > max_frames:
        return xyz[:max_frames]
    else:
        pad_len = max_frames - n_frames
        pad = np.zeros((pad_len, xyz.shape[1], xyz.shape[2]), dtype=xyz.dtype)
        return np.concatenate([xyz, pad], axis=0)


# ============================
# Load TFLite model + label map
# ============================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_index = input_details["index"]
output_index = output_details["index"]
input_shape = input_details["shape"]
input_rank = len(input_shape)

with open(SIGN_MAP_PATH, "r") as f:
    SIGN2IDX = json.load(f)
IDX2SIGN = {v: k for k, v in SIGN2IDX.items()}


def predict_sign_from_parquet(pq_path: str) -> str:
    xyz = load_relevant_data_subset(pq_path)
    xyz = pad_or_trim(xyz, MAX_FRAMES)  # (32, 543, 3)

    # Chuẩn input theo rank của model
    if input_rank == 3:
        model_input = xyz.astype(np.float32)                # (32, 543, 3)
    else:
        model_input = np.expand_dims(xyz, axis=0).astype(np.float32)  # (1, 32, 543, 3)

    interpreter.set_tensor(input_index, model_input)
    interpreter.invoke()
    outputs = interpreter.get_tensor(output_index)[0]  # (num_classes,)
    idx = int(np.argmax(outputs))
    sign = IDX2SIGN.get(idx, f"UNKNOWN_{idx}")
    return sign


if __name__ == "__main__":
    # 👉 ĐỔI đường dẫn này thành 1 file parquet có thật trong train_landmark_files
    pq_file = "/home/lananh/GISLR/train_landmark_files/2044/635217.parquet"

    if not os.path.exists(pq_file):
        raise FileNotFoundError(f"Không tìm thấy file: {pq_file}")

    start = time.time()
    sign = predict_sign_from_parquet(pq_file)
    end = time.time()

    print(f"File: {pq_file}")
    print(f"Predicted sign: {sign}")
    print(f"Inference time: {end - start:.3f} s")
