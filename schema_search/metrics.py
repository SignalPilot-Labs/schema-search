import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def dot_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return -np.linalg.norm(a[:, None] - b[None, :], axis=-1)


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return -np.sum(np.abs(a[:, None] - b[None, :]), axis=-1)


METRICS = {
    "cosine": cosine_similarity,
    "dot": dot_product,
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
}


def get_metric(name: str):
    if name not in METRICS:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRICS.keys())}")
    return METRICS[name]
