"""Reconstruct how each final image maps back through the curation steps.

Chain (all verified against the stored metadata, no model is run):

    aged_images2/<ethnicity>/<age_group>/imgNNN.png      (manually curated, relabelled)
      -> consolidated  ``imgNNN.png`` + ``id``            (consolidate_dataset.py)
      -> CodeFormer    same filename, restored            (codeFormed_restored)
      -> shuffled      ``df.sample(frac=1, random_state=42)``, renamed imgNNN.png (shuffle.py)
      -> final         ``imgNNNN.png`` (4-digit)          (same order; 11 unknown genders fixed)

Known gap: the id of the *base* SDXL image is not recoverable from filenames,
because the curation app renumbered each age-group folder. See README.
"""
from __future__ import annotations

import re

import pandas as pd

LABEL_COLUMNS = ["ethnicity", "age_group", "gender"]
SHUFFLE_SEED = 42


class LineageError(RuntimeError):
    """Raised when the stored metadata files are not consistent with each other."""


def image_number(filename: str) -> int:
    """'img0007.png' / 'img007.png' -> 7."""
    match = re.fullmatch(r"img(\d+)\.\w+", filename)
    if not match:
        raise ValueError(f"unexpected filename: {filename!r}")
    return int(match.group(1))


def restored_name(index: int) -> str:
    """Name given by shuffle.py to the i-th (1-based) shuffled image."""
    return f"img{index:03d}.png"


def final_name(index: int) -> str:
    return f"img{index:04d}.png"


def _same_labels(a: pd.DataFrame, b: pd.DataFrame, columns: list[str]) -> pd.Series:
    return (a[columns].fillna("NA").values == b[columns].fillna("NA").values).all(axis=1)


def build_lineage(
    consolidated: pd.DataFrame,
    restored: pd.DataFrame,
    final: pd.DataFrame,
    train: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
    seed: int = SHUFFLE_SEED,
) -> pd.DataFrame:
    """Return one row per final image with its upstream identifiers.

    ``consolidated`` is the pre-shuffle table (has ``id`` = index inside its
    ``<ethnicity>/<age_group>`` curation folder), ``restored`` the shuffled
    table written by ``shuffle.py`` and ``final`` the submitted dataset table.
    """
    if not (len(consolidated) == len(restored) == len(final)):
        raise LineageError("tables have different lengths")

    shuffled = consolidated.sample(frac=1, random_state=seed).reset_index(drop=True)
    if not _same_labels(shuffled, restored, ["ethnicity", "age_group"]).all():
        raise LineageError("seeded shuffle does not reproduce the restored order")
    if not _same_labels(restored, final, ["ethnicity", "age_group"]).all():
        raise LineageError("final table is not in restored order")

    out = pd.DataFrame(
        {
            "final_filename": [final_name(i + 1) for i in range(len(final))],
            "restored_filename": restored["filename"].values,
            "consolidated_filename": shuffled["filename"].values,
            "curation_folder_id": shuffled["id"].values,
            "ethnicity": final["ethnicity"].values,
            "age_group": final["age_group"].values,
            "gender": final["gender"].values,
        }
    )
    expected = [restored_name(i + 1) for i in range(len(final))]
    if out["restored_filename"].tolist() != expected:
        raise LineageError("restored filenames are not sequential")

    out["gender_corrected"] = (
        restored["gender"].values != final["gender"].values
    ) & (restored["gender"].values == "Unknown")

    if train is not None and test is not None:
        split = {f: "train" for f in train["filename"]} | {f: "test" for f in test["filename"]}
        out["split"] = out["restored_filename"].map(split)
    return out
