import tensorflow as tf

H5_MODEL = "/home/lananh/GISLR/asl-transformer/asl_transformer_20.h5"
TFLITE_MODEL = "/home/lananh/GISLR/asl-transformer/asl_transformer_20.tflite"

if __name__ == "__main__":
    # 1. Load Keras model
    print("Loading Keras model:", H5_MODEL)
    model = tf.keras.models.load_model(H5_MODEL)

    # 2. Tạo converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Dùng cả SELECT_TF_OPS cho chắc (MultiHeadAttention đôi khi cần)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    # Với transformer (không dùng LSTM/TensorList) thật ra không bắt buộc,
    # nhưng để đồng bộ code:
    converter._experimental_lower_tensor_list_ops = False

    # 3. Convert
    print("Converting to TFLite...")
    tflite_model = converter.convert()

    # 4. Save .tflite
    with open(TFLITE_MODEL, "wb") as f:
        f.write(tflite_model)

    print("Saved TFLite model to:", TFLITE_MODEL)
