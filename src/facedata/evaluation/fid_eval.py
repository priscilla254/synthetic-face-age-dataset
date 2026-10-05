"""Fréchet Inception Distance.

The dissertation scores (141.18 vs FairFace, 156.58 vs UTKFace) were produced with
the Keras ``InceptionV3(imagenet, pooling='avg')`` features of the original script
(``legacy_as_run/evaluation/3.realism_with_FID.py``). They are *not* directly
comparable with FID values computed using the standard pytorch-fid / TF-Inception
weights. This module only restructures that computation; it does not re-run it.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """||mu1-mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^(1/2)), using eigenvalues only.

    Tr((S1 S2)^(1/2)) equals the sum of square roots of the eigenvalues of S1 S2
    (non-negative for PSD inputs), so no matrix square root (scipy) is needed.
    """
    diff = mu1 - mu2
    eigvals = np.linalg.eigvals(sigma1 @ sigma2)
    tr_covmean = np.sqrt(np.clip(eigvals.real, 0.0, None)).sum()
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def feature_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return features.mean(axis=0), np.cov(features, rowvar=False)


class FIDEvaluator:
    """FID between two image sets given a feature extractor ``path -> vector``."""

    def __init__(self, extractor: Callable[[Path], np.ndarray]):
        self.extractor = extractor

    def features(self, paths: Sequence[Path]) -> np.ndarray:
        feats = [np.asarray(self.extractor(Path(p))).ravel() for p in paths]
        if not feats:
            raise ValueError("no images given")
        return np.vstack(feats)

    def score(self, real_paths: Sequence[Path], synthetic_paths: Sequence[Path]) -> float:
        mu_r, s_r = feature_statistics(self.features(real_paths))
        mu_s, s_s = feature_statistics(self.features(synthetic_paths))
        return frechet_distance(mu_r, s_r, mu_s, s_s)


def keras_inception_extractor() -> Callable[[Path], np.ndarray]:
    """Feature extractor matching the original script (needs TensorFlow)."""
    from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
    from tensorflow.keras.preprocessing import image

    model = InceptionV3(include_top=False, weights="imagenet", pooling="avg")

    def extract(path: Path) -> np.ndarray:
        img = image.load_img(path, target_size=(299, 299))
        arr = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
        return model.predict(arr, verbose=0)[0]

    return extract
