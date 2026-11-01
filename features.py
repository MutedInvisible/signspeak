"""Turn MediaPipe hand landmarks into a more discriminative feature vector."""

from __future__ import annotations

import numpy as np

LANDMARKS_PER_HAND = 21
COORDS = 3
HANDS = 2

# Original 126 + extra distances & angles
FEATURE_SIZE = 126 + 30  # keep it reasonable

# Key landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17


def _points(hand) -> np.ndarray:
    pts = hand.landmark if hasattr(hand, "landmark") else hand
    return np.array([[lm.x, lm.y, lm.z] for lm in pts], dtype=np.float32)


def _normalize_hand(xyz: np.ndarray) -> np.ndarray:
    """Center on wrist and scale by palm size."""
    wrist = xyz[0].copy()
    centered = xyz - wrist
    scale = np.linalg.norm(centered[MIDDLE_MCP])
    if scale < 1e-6:
        scale = np.max(np.linalg.norm(centered, axis=1)) or 1.0
    return centered / scale


def _hand_label(hand_info, index: int) -> str:
    if not hand_info:
        return "Right"
    item = hand_info[index]
    if hasattr(item, "classification"):
        return item.classification[0].label
    if isinstance(item, (list, tuple)):
        cat = item[0]
        return getattr(cat, "category_name", None) or getattr(cat, "display_name", "Right")
    return getattr(item, "category_name", None) or "Right"


def _extra_features(xyz: np.ndarray) -> np.ndarray:
    """Distances between fingertips + a few angles. xyz is already normalized."""
    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    dists = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dists.append(np.linalg.norm(xyz[tips[i]] - xyz[tips[j]]))

    # Simple angle proxies (dot products of bone vectors)
    def bone(a, b):
        return xyz[b] - xyz[a]

    v1 = bone(WRIST, INDEX_MCP)
    v2 = bone(WRIST, MIDDLE_MCP)
    v3 = bone(WRIST, PINKY_MCP)
    v4 = bone(INDEX_MCP, INDEX_TIP)
    v5 = bone(MIDDLE_MCP, MIDDLE_TIP)

    def cos_sim(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    angles = [
        cos_sim(v1, v2),
        cos_sim(v2, v3),
        cos_sim(v1, v4),
        cos_sim(v2, v5),
        cos_sim(v4, v5),
    ]

    return np.array(dists + angles, dtype=np.float32)


def landmarks_to_features(multi_hand_landmarks, handedness) -> np.ndarray | None:
    """Build feature vector: left hand then right hand (+ extras). Missing hands are zeros."""
    if not multi_hand_landmarks:
        return None

    left = np.zeros(LANDMARKS_PER_HAND * COORDS, dtype=np.float32)
    right = np.zeros(LANDMARKS_PER_HAND * COORDS, dtype=np.float32)
    left_extra = np.zeros(15, dtype=np.float32)   # 10 dists + 5 angles
    right_extra = np.zeros(15, dtype=np.float32)

    for i, landmarks in enumerate(multi_hand_landmarks):
        xyz = _normalize_hand(_points(landmarks))
        vec = xyz.reshape(-1)
        extra = _extra_features(xyz)
        label = _hand_label(handedness, i)
        if str(label).lower().startswith("left"):
            left = vec
            left_extra = extra
        else:
            right = vec
            right_extra = extra

    return np.concatenate([left, right, left_extra, right_extra]).astype(np.float32)


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)