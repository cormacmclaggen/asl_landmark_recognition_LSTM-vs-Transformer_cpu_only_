import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf

ROWS_PER_FRAME = 543
MAX_FRAMES = 32

TFLITE_MODEL = "/home/lananh/GISLR/asl-transformer/asl_transformer_20.tflite"
LABEL_JSON = "/home/lananh/GISLR/asl-transformer/asl_transformer_20_sign_to_idx.json"


def load_parquet_for_inference(path):
    df = pd.read_parquet(path, columns=["x", "y", "z"])
    df = df.values.reshape(-1, ROWS_PER_FRAME, 3).astype(np.float32)

    # Fix length
    if df.shape[0] > MAX_FRAMES:
        df = df[:MAX_FRAMES]
    else:
        pad = np.zeros((MAX_FRAMES - df.shape[0], ROWS_PER_FRAME, 3), dtype=np.float32)
        df = np.concatenate([df, pad], axis=0)

    # Normalize same as during training (per sample)
    mean = np.nanmean(df, axis=(0, 1), keepdims=True)
    std = np.nanstd(df, axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1e-6
    df = (df - mean) / std
    df = np.nan_to_num(df, nan=0.0, posinf=0.0, neginf=0.0)

    return df.astype(np.float32)   # (32, 543, 3)


if __name__ == "__main__":
    # 👉 CHANGE this path to a real parquet file in train_landmark_files
    pq_file = "/home/lananh/GISLR/train_landmark_files/16069/3380002.parquet"

    if not os.path.exists(pq_file):
        raise FileNotFoundError(f"File not found: {pq_file}")

    # Load label map
    with open(LABEL_JSON, "r") as f:
        sign2idx = json.load(f)
    idx2sign = {v: k for k, v in sign2idx.items()}

    # Load TFLite
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_index = input_details["index"]
    output_index = output_details["index"]
    input_rank = len(input_details["shape"])
    print("TFLite input shape:", input_details["shape"])

    # Prepare input
    xyz = load_parquet_for_inference(pq_file)  # (32, 543, 3)

    if input_rank == 3:
        model_input = xyz                        # (32, 543, 3)
    else:
        model_input = np.expand_dims(xyz, 0)     # (1, 32, 543, 3)

    # Inference
    start = time.time()
    interpreter.set_tensor(input_index, model_input.astype(np.float32))
    interpreter.invoke()
    outputs = interpreter.get_tensor(output_index)[0]  # (num_classes,)
    end = time.time()

    idx = int(np.argmax(outputs))
    sign = idx2sign.get(idx, f"UNKNOWN_{idx}")

    print(f"File: {pq_file}")
    print(f"Predicted sign: {sign}")
    print(f"Inference time: {end - start:.3f} s")
