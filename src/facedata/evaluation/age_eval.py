"""Age-group check with InsightFace predictions (reads the stored CSV)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from facedata.dataset.age_bins import AGE_GROUPS, age_to_group, group_index, normalise_group


@dataclass
class AgeResult:
    accuracy: float
    adjacent_accuracy: float
    n: int
    per_group: pd.DataFrame      # n, correct, accuracy, adjacent_accuracy, mean_signed_bin_error
    confusion: pd.DataFrame      # rows: intended group, columns: predicted group


class AgeEstimationEvaluator:
    """Bin predicted ages and compare with the intended age group.

    ``adjacent_accuracy`` counts a prediction as acceptable when it lands in the
    intended bin or the neighbouring one; ``mean_signed_bin_error`` > 0 means the
    model over-estimates age (predicts an older bin than intended).
    """

    @staticmethod
    def load_predictions(path) -> pd.DataFrame:
        df = pd.read_csv(path)
        # InsightFace crops are named 'img0001_00.png'
        df["filename"] = df["filename"].str.replace(r"_\d+(?=\.\w+$)", "", regex=True)
        df["predicted_group"] = df["age"].map(age_to_group)
        return df[["filename", "age", "predicted_group"]]

    def evaluate(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> AgeResult:
        gt = ground_truth[["filename", "age_group"]].rename(columns={"age_group": "true_group"})
        gt["true_group"] = gt["true_group"].map(normalise_group)
        merged = predictions.merge(gt, on="filename")
        merged["true_idx"] = merged["true_group"].map(group_index)
        merged["pred_idx"] = merged["predicted_group"].map(group_index)
        # predictions outside the dataset range are placed just outside the scale
        merged.loc[merged["predicted_group"] == "0-12", "pred_idx"] = -1
        merged = merged.dropna(subset=["true_idx", "pred_idx"])
        merged["error"] = merged["pred_idx"] - merged["true_idx"]
        merged["exact"] = merged["error"] == 0
        merged["adjacent"] = merged["error"].abs() <= 1

        rows = []
        for group in AGE_GROUPS:
            sub = merged[merged["true_group"] == group]
            rows.append(
                {
                    "age_group": group,
                    "n": len(sub),
                    "correct": int(sub["exact"].sum()),
                    "accuracy": sub["exact"].mean() if len(sub) else float("nan"),
                    "adjacent_accuracy": sub["adjacent"].mean() if len(sub) else float("nan"),
                    "mean_signed_bin_error": sub["error"].mean() if len(sub) else float("nan"),
                }
            )
        confusion = pd.crosstab(merged["true_group"], merged["predicted_group"])
        confusion = confusion.reindex(index=list(AGE_GROUPS))
        return AgeResult(
            accuracy=float(merged["exact"].mean()),
            adjacent_accuracy=float(merged["adjacent"].mean()),
            n=len(merged),
            per_group=pd.DataFrame(rows).set_index("age_group"),
            confusion=confusion.fillna(0).astype(int),
        )
