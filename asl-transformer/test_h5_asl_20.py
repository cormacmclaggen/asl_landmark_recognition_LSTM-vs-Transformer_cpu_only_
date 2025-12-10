#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Test Keras ASL Transformer 20 model for NaN/Inf outputs
import numpy as np
import tensorflow as tf

KERAS_MODEL_PATH = "/home/lananh/GISLR/asl-transformer/asl_transformer_20.h5"  # ĐỔI CHO ĐÚNG

def main():
    print("[INFO] Loading Keras model:", KERAS_MODEL_PATH)
    model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)

    # Giả sử input shape (1, 32, 543, 3)
    input_shape = (1, 32, 543, 3)
    x = np.random.randn(*input_shape).astype(np.float32)

    y = model(x, training=False).numpy()

    print("[INFO] output shape:", y.shape)
    print("[INFO] output min:", np.min(y), "max:", np.max(y))
    print("[INFO] any NaN ?", np.isnan(y).any())
    print("[INFO] any Inf ?", np.isinf(y).any())
    print("[INFO] first 10:", y.flatten()[:10])

if __name__ == "__main__":
    main()
