"""Rebuild data/metadata/lineage.csv from the stored metadata tables."""
from pathlib import Path

import pandas as pd

from facedata.dataset.lineage import build_lineage

M = Path(__file__).resolve().parents[1] / "data" / "metadata"

lineage = build_lineage(
    consolidated=pd.read_csv(M / "stage3_consolidated_pre_codeformer.csv"),
    restored=pd.read_csv(M / "stage4_restored_shuffled_all.csv"),
    final=pd.read_csv(M / "final_dataset_original_labels.csv"),
    train=pd.read_csv(M / "stage4_restored_shuffled_train.csv"),
    test=pd.read_csv(M / "stage4_restored_shuffled_test.csv"),
)
lineage.to_csv(M / "lineage.csv", index=False)
print(f"wrote {len(lineage)} rows -> {M / 'lineage.csv'}")
print(lineage["split"].value_counts().to_dict())
print("gender corrections:", int(lineage["gender_corrected"].sum()))
