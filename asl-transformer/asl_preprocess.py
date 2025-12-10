import numpy as np

POSE_LANDMARKS = 33
FACE_LANDMARKS = 468
HAND_LANDMARKS = 21

def preprocess_landmarks_543x3(landmarks_543x3: np.ndarray) -> np.ndarray:
    """
    landmarks_543x3: (543, 3) với thứ tự:
    [pose(33), face(468), left hand(21), right hand(21)]
    """
    assert landmarks_543x3.shape == (543, 3)

    # 1) dịch về quanh mũi
    nose = landmarks_543x3[0, :].copy()
    centered = landmarks_543x3 - nose

    # 2) scale theo khoảng cách hai vai
    left_sh = landmarks_543x3[11, :]
    right_sh = landmarks_543x3[12, :]
    body_size = np.linalg.norm(left_sh - right_sh) + 1e-6
    normalized = centered / body_size

    # 3) clip
    normalized = np.clip(normalized, -3.0, 3.0)

    return normalized.astype(np.float32)
