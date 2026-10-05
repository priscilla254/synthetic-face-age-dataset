"""Age-group definitions shared by generation, curation and evaluation."""
from __future__ import annotations

import math

# Order matters: it is the ordinal scale used for "off by one bin" metrics.
AGE_GROUPS: tuple[str, ...] = ("13-17", "18-20", "21-25", "26-40", "41plus")

# AgeTransGAN output tags -> dataset age groups (as used in the original sorting step).
AGE_TAG_TO_GROUP: dict[str, str] = {
    "age1": "13-17",
    "age2": "18-20",
    "age3": "21-25",
    "age4": "26-40",
    "age5": "41plus",
}

UNDER_RANGE = "0-12"   # predicted age below the dataset's youngest bin
UNKNOWN = "unknown"    # missing prediction


def normalise_group(label: str) -> str:
    """Map the '41+' spelling used by the evaluation notebooks to '41plus'."""
    return "41plus" if label == "41+" else label


def age_to_group(age: float | None) -> str:
    """Bin a numeric age into the dataset's age groups.

    Mirrors the InsightFace binning used in the dissertation: ages under 13 fall
    into ``'0-12'`` and missing values into ``'unknown'``.
    """
    if age is None or (isinstance(age, float) and math.isnan(age)):
        return UNKNOWN
    age = int(age)
    if age < 13:
        return UNDER_RANGE
    if age <= 17:
        return "13-17"
    if age <= 20:
        return "18-20"
    if age <= 25:
        return "21-25"
    if age <= 40:
        return "26-40"
    return "41plus"


def group_index(label: str) -> int | None:
    """Ordinal position of an age group (None for out-of-range / unknown labels)."""
    label = normalise_group(label)
    return AGE_GROUPS.index(label) if label in AGE_GROUPS else None
