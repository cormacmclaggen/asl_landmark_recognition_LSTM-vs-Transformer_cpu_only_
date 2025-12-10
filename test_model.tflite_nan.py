#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

MODEL_PATH = "/home/lananh/GISLR/model.tflite"   # sửa nếu m để chỗ khác

def main():
    print("[INFO] Loading TFLite model:", MODEL_PATH)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_index = input_details["index"]
    output_index = output_details["index"]

    input_shape = input_details["shape"]
    input_dtype = input_details["dtype"]

    print("[INFO] input_shape:", input_shape, "dtype:", input_dtype)

    # Tạo dữ liệu random nhỏ nhỏ để tránh overflow
    # ví dụ N(0, 1)
    x = np.random.randn(*input_shape).astype(np.float32)

    # Nếu model mong chờ float16 / int8 thì cast sang đúng kiểu
    x = x.astype(input_dtype)

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()

    y = interpreter.get_tensor(output_index)[0]  # (num_classes, ...) usually

    print("[INFO] output shape:", y.shape)
    print("[INFO] output min:", np.min(y), "max:", np.max(y))

    # Kiểm tra NaN / Inf
    print("[INFO] any NaN in output? ", np.isnan(y).any())
    print("[INFO] any Inf in output? ", np.isinf(y).any())

    # In thử 10 giá trị đầu
    flat = y.flatten()
    print("[INFO] first 10 values:", flat[:10])

if __name__ == "__main__":
    main()
