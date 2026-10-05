"""Summaries of the human rating study (24 participants, 1-5 scales)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CRITERIA = ["realism", "age_appropriateness", "ethnic_consistency"]


class HumanEvalAnalyser:
    def __init__(self, df: pd.DataFrame):
        missing = [c for c in CRITERIA + ["filename", "ethnicity", "age_group"] if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        df = df.copy()
        df[CRITERIA] = df[CRITERIA].apply(pd.to_numeric, errors="coerce")
        self.df = df.dropna(subset=CRITERIA)

    @classmethod
    def from_csv(cls, path) -> HumanEvalAnalyser:
        return cls(pd.read_csv(path))

    @staticmethod
    def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
        rng = np.random.default_rng(seed)
        values = np.asarray(values, dtype=float)
        means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
        return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))

    def summary(self, by: str | None = None) -> pd.DataFrame:
        """Mean, 95% bootstrap CI and n per criterion (optionally per group)."""
        groups = [(None, self.df)] if by is None else list(self.df.groupby(by))
        rows = []
        for name, sub in groups:
            for criterion in CRITERIA:
                lo, hi = self.bootstrap_ci(sub[criterion].values)
                row = {"criterion": criterion, "n_ratings": len(sub),
                       "mean": sub[criterion].mean(), "ci95_low": lo, "ci95_high": hi}
                if by is not None:
                    row = {by: name} | row
                rows.append(row)
        return pd.DataFrame(rows)

    def correlation(self) -> pd.DataFrame:
        return self.df[CRITERIA].corr()

    def worst_images(self, n: int = 10) -> pd.DataFrame:
        per_image = self.df.groupby(["filename", "ethnicity"])[CRITERIA].mean().reset_index()
        return per_image.sort_values(CRITERIA).head(n)
