#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run TFLite inference on GISLR parquet files,
same logic as the Jupyter notebook.

Usage example:
    (mp_env) python run_tflite_parquet.py \
        /home/lananh/GISLR/data_parquet/shhh-0.parquet \
        /home/lananh/GISLR/data_parquet/bird-0.parquet \
        /home/lananh/GISLR/data_parquet/duck-0.parquet
"""

import argparse
import os

import numpy as np
import pandas as pd

# If you used tflite_runtime in the notebook:
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback to tensorflow.lite if tflite_runtime is not available
    import tensorflow.lite as tflite

# ==============================
# CONFIG – UPDATE PATHS
# ==============================

MODEL_PATH = "/home/lananh/GISLR/model.tflite"
TRAIN_CSV_PATH = "/home/lananh/GISLR/train.csv"

ROWS_PER_FRAME = 543   # number of landmarks per frame (fixed in GISLR)
DATA_COLUMNS = ["x", "y", "z"]


# ==============================
# 1. READ PARQUET → (n_frames, 543, 3)
# ==============================

def load_relevant_data_subset(pq_path: str) -> np.ndarray:
    """
    Read a GISLR parquet (x, y, z columns) and reshape to
        (n_frames, ROWS_PER_FRAME, 3)
    exactly like the notebook code.
    """
    data = pd.read_parquet(pq_path, columns=DATA_COLUMNS)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(DATA_COLUMNS))
    return data.astype(np.float32)


# ==============================
# 2. LOAD MODEL + CREATE prediction_fn
# ==============================

def load_tflite_model(model_path: str):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    sigs = interpreter.get_signature_list()
    if "serving_default" not in sigs:
        raise RuntimeError(
            f"Model does not have a 'serving_default' signature. "
            f"Available signatures: {list(sigs.keys())}"
        )

    sig = sigs["serving_default"]
    # Get the actual input/output names, e.g. 'input_1', 'outputs'
    input_keys = list(sig["inputs"].keys())
    output_keys = list(sig["outputs"].keys())
    if len(input_keys) != 1 or len(output_keys) != 1:
        raise RuntimeError(f"Unexpected signature structure: {sig}")

    input_key = input_keys[0]
    output_key = output_keys[0]

    print("[INFO] Signature 'serving_default':")
    print("       input :", input_key)
    print("       output:", output_key)

    prediction_fn = interpreter.get_signature_runner("serving_default")
    return prediction_fn, input_key, output_key


# ==============================
# 3. LOAD LABEL MAP from train.csv
# ==============================

def load_label_mapping(train_csv_path: str):
    train = pd.read_csv(train_csv_path)
    # ordinal encode
    train["sign_ord"] = train["sign"].astype("category").cat.codes

    # dict sign -> ord, ord -> sign
    sign2ord = (
        train[["sign", "sign_ord"]]
        .drop_duplicates()
        .set_index("sign")["sign_ord"]
        .to_dict()
    )
    ord2sign = (
        train[["sign_ord", "sign"]]
        .drop_duplicates()
        .set_index("sign_ord")["sign"]
        .to_dict()
    )
    return sign2ord, ord2sign


# ==============================
# 4. INFERENCE ON ONE PARQUET FILE
# ==============================

def get_prediction(prediction_fn, input_key, output_key, ord2sign, pq_file: str):
    """
    Run inference on a parquet file and print the result.
    """
    if not os.path.exists(pq_file):
        print(f"[ERROR] File does not exist: {pq_file}")
        return

    xyz_np = load_relevant_data_subset(pq_file)  # (n_frames, 543, 3)

    # Feed data with the correct input_key (e.g. 'inputs', 'input_1', ...)
    prediction = prediction_fn(**{input_key: xyz_np})

    # Get the correct output_key (e.g. 'outputs', 'output_0', ...)
    outputs = prediction[output_key]
    outputs = np.asarray(outputs, dtype=np.float32)

    if outputs.ndim > 1:
        # In case there is a batch dimension
        outputs = outputs[0]

    # Light softmax (if the model doesn't already include it)
    logits = outputs - np.max(outputs)
    exp = np.exp(logits)
    probs = exp / (np.sum(exp) + 1e-8)

    pred_idx = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])
    sign = ord2sign.get(pred_idx, f"IDX_{pred_idx}")

    print(
        f"PREDICTED SIGN: {sign} [{pred_idx}], "
        f"CONFIDENCE: {pred_conf:0.4f}    (file: {os.path.basename(pq_file)})"
    )


# ==============================
# 5. MAIN
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description="Run TFLite inference on GISLR parquet file(s)"
    )
    parser.add_argument(
        "parquet_files",
        nargs="+",
        help="Path to one or more .parquet files to run prediction on",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading TFLite model from: {MODEL_PATH}")
    prediction_fn, input_key, output_key = load_tflite_model(MODEL_PATH)

    print(f"[INFO] Loading label mapping from: {TRAIN_CSV_PATH}")
    _, ord2sign = load_label_mapping(TRAIN_CSV_PATH)

    print("\n========== INFERENCE ==========")
    for pq in args.parquet_files:
        get_prediction(prediction_fn, input_key, output_key, ord2sign, pq)


if __name__ == "__main__":
    main()
