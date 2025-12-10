#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Test TFLite ASL Transformer 20 model for NaN/Inf outputs

import numpy as np
import tensorflow as tf

TFLITE_MODEL_PATH = "/home/lananh/GISLR/asl-transformer/asl_transformer_20.tflite"

def main():
    print("[INFO] Loading TFLite model:", TFLITE_MODEL_PATH)
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_index = input_details["index"]
    output_index = output_details["index"]

    input_shape = input_details["shape"]
    input_dtype = input_details["dtype"]

    print("[INFO] input_shape:", input_shape, "dtype:", input_dtype)

    # random input nhỏ nhỏ
    x = np.random.randn(*input_shape).astype(np.float32)
    x = x.astype(input_dtype)

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()

    y = interpreter.get_tensor(output_index)[0]

    print("[INFO] output shape:", y.shape)
    print("[INFO] output min:", np.min(y), "max:", np.max(y))
    print("[INFO] any NaN ?", np.isnan(y).any())
    print("[INFO] any Inf ?", np.isinf(y).any())
    print("[INFO] first 10:", y.flatten()[:10])

if __name__ == "__main__":
    main()
