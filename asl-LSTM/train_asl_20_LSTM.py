import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time

# ======================================
# CONFIG
# ======================================
ROWS_PER_FRAME = 543
MAX_FRAMES = 32

TRAIN_CSV = "train.csv"
DATA_FOLDER = "train_landmark_files"   # folder chứa các parquet

OUTPUT_MODEL = "asl_20_LSTM.h5"
OUTPUT_JSON = "asl_20_LSTM_sign_to_idx.json"

tf.random.set_seed(42)
np.random.seed(42)


# ======================================
# 1. Load and normalize parquet data
# ======================================

def resolve_parquet_path(raw_path: str) -> str:
    """
    Cố gắng tìm đường dẫn thật:
    - Nếu raw_path tồn tại -> dùng luôn
    - Nếu không, thử thêm 'train_landmark_files/' ở trước
    """
    raw_path = raw_path.strip()

    if os.path.exists(raw_path):
        return raw_path

    candidate = os.path.join(DATA_FOLDER, raw_path)
    if os.path.exists(candidate):
        return candidate

    # Nếu vẫn không tồn tại thì trả về raw_path (sẽ bị skip sau)
    return raw_path


def load_parquet(path):
    df = pd.read_parquet(path, columns=["x", "y", "z"])
    df = df.values.reshape(-1, ROWS_PER_FRAME, 3).astype(np.float32)

    # Fix frame length
    if df.shape[0] > MAX_FRAMES:
        df = df[:MAX_FRAMES]
    else:
        pad = np.zeros((MAX_FRAMES - df.shape[0], ROWS_PER_FRAME, 3), dtype=np.float32)
        df = np.concatenate([df, pad], axis=0)

    # Normalize (per sample)
    mean = np.nanmean(df, axis=(0, 1), keepdims=True)
    std = np.nanstd(df, axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1e-6  # tránh chia cho 0
    df = (df - mean) / std

    # Loại NaN / inf nếu vẫn còn
    df = np.nan_to_num(df, nan=0.0, posinf=0.0, neginf=0.0)

    return df.astype(np.float32)


# ======================================
# 2. Build dataset from TOP frequent signs
# ======================================

def build_dataset(top_k=20, samples_per_sign=300):
    csv = pd.read_csv(TRAIN_CSV)

    # Find top-k frequent labels
    top_signs = csv["sign"].value_counts().head(top_k).index.tolist()
    print("Selected signs:", top_signs)

    sign_to_idx = {s: i for i, s in enumerate(top_signs)}

    X, y = [], []
    skipped = 0

    for sign in top_signs:
        subset = csv[csv["sign"] == sign]

        # limit samples
        subset = subset.sample(
            min(samples_per_sign, len(subset)),
            random_state=42
        )

        for _, row in subset.iterrows():
            raw_path = row["path"]
            pq_path = resolve_parquet_path(raw_path)

            if not os.path.exists(pq_path):
                # print("Missing file:", pq_path)
                skipped += 1
                continue

            try:
                X.append(load_parquet(pq_path))
                y.append(sign_to_idx[sign])
            except Exception as e:
                print("Error loading:", pq_path, "→", e)
                skipped += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print("Dataset shape:", X.shape, y.shape)
    print("Skipped samples:", skipped)
    print("Any NaN in X?", np.isnan(X).any())
    print("Any inf in X?", np.isinf(X).any())

    if len(X) == 0:
        raise ValueError("No samples loaded. Check paths / parquet folder.")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(sign_to_idx, f)
    print("Saved label map to:", OUTPUT_JSON)

    return X, y, sign_to_idx


# ======================================
# 3. Build the model
# ======================================

def build_model(num_classes):
    inp = tf.keras.Input(shape=(MAX_FRAMES, ROWS_PER_FRAME, 3))

    x = tf.keras.layers.Reshape((MAX_FRAMES, ROWS_PER_FRAME * 3))(inp)
    x = tf.keras.layers.Masking(mask_value=0.0)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True)
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256)
    )(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ======================================
# 4. Train
# ======================================

if __name__ == "__main__":
    X, y, labels = build_dataset(top_k=20, samples_per_sign=300)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    print("Train size:", X_train.shape, "Val size:", X_val.shape)

    model = build_model(num_classes=len(labels))
    model.summary()

    epochs = 20
    batch_size = 32

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    end = time.time()

    total_time = end - start
    print(f"LSTM training time for {epochs} epochs: {total_time:.2f} seconds")
    print(f"≈ {total_time / epochs:.2f} seconds per epoch")

    model.save(OUTPUT_MODEL)
    print("Saved model:", OUTPUT_MODEL)
    print("Saved labels:", OUTPUT_JSON)

val_loss, val_acc = model.evaluate(X_val, y_val, batch_size=32)
print(f"Final val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}")
