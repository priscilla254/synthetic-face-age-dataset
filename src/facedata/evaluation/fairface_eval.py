"""Ethnicity-label check with FairFace predictions (reads the stored CSV)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Mapping as saved in the dissertation notebook. NOTE: it has no entry for the
# sub-Saharan labels used in the original metadata (East/West/South African,
# Ethiopian), so running it drops 329 prediction rows and gives 0.66 overall.
# The accuracy reported in the dissertation (0.70) is reproduced by
# EXTENDED_MAPPING below, so that is the default here.
NOTEBOOK_MAPPING: dict[str, str] = {
    "Southern European": "White",
    "Northern European": "White",
    "Chinese": "East Asian",
    "Filipino": "Southeast Asian",
    "Indian": "Indian",
    "Arabic": "Middle Eastern",
    "African": "Black",
    "Black Caribbean": "Black",
    "Native American": "Latino_Hispanic",
    "Hawaiian": "Southeast Asian",
    "Southeast Asian": "Southeast Asian",
    "Brazilian": "Latino_Hispanic",
    "Indigenous Australian": "Southeast Asian",
}

# Same mapping with the sub-Saharan labels included (reproduces the reported 0.70).
EXTENDED_MAPPING: dict[str, str] = NOTEBOOK_MAPPING | {
    "East African": "Black",
    "West African": "Black",
    "South African": "Black",
    "Ethiopian": "Black",
}


@dataclass
class FairFaceResult:
    accuracy: float
    n_rows: int
    n_unmapped_dropped: int
    per_group_accuracy: pd.Series
    confusion: pd.DataFrame


class FairFaceEvaluator:
    """Compare intended ethnicity (mapped to FairFace classes) with FairFace output.

    Parameters
    ----------
    mapping: dataset ethnicity -> FairFace ``race`` class (default: EXTENDED_MAPPING).
    """

    def __init__(self, mapping: dict[str, str] | None = None):
        self.mapping = dict(EXTENDED_MAPPING if mapping is None else mapping)

    @staticmethod
    def load_predictions(path) -> pd.DataFrame:
        df = pd.read_csv(path)
        # 'detected_faces/img0752_face0.png' -> 'img0752.png'
        df["filename"] = df["face_name_align"].map(
            lambda p: p.split("/")[-1].split("_")[0] + ".png"
        )
        return df

    def evaluate(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> FairFaceResult:
        gt = ground_truth.copy()
        gt["expected"] = gt["ethnicity"].map(self.mapping)
        merged = predictions.merge(gt, on="filename")
        unmapped = int(merged["expected"].isna().sum())
        merged = merged.dropna(subset=["expected"])
        correct = merged["race"] == merged["expected"]
        per_group = correct.groupby(merged["expected"]).mean().sort_values(ascending=False)
        confusion = pd.crosstab(merged["expected"], merged["race"])
        return FairFaceResult(
            accuracy=float(correct.mean()),
            n_rows=len(merged),
            n_unmapped_dropped=unmapped,
            per_group_accuracy=per_group,
            confusion=confusion,
        )
