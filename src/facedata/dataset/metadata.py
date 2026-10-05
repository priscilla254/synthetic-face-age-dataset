"""Metadata table handling: loading, validation and distribution summaries."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .age_bins import AGE_GROUPS

REQUIRED_COLUMNS = ("filename", "ethnicity", "age_group", "gender")


class MetadataError(ValueError):
    """Raised when a metadata table does not meet the dataset schema."""


@dataclass
class MetadataStore:
    """A validated view over a dataset metadata CSV."""

    df: pd.DataFrame

    @classmethod
    def from_csv(cls, path: str | Path, validate: bool = True) -> MetadataStore:
        store = cls(pd.read_csv(path))
        if validate:
            store.validate()
        return store

    def validate(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise MetadataError(f"missing columns: {missing}")
        if self.df["filename"].duplicated().any():
            dupes = self.df.loc[self.df["filename"].duplicated(), "filename"].tolist()
            raise MetadataError(f"duplicate filenames: {dupes[:5]}")
        bad = sorted(set(self.df["age_group"].dropna()) - set(AGE_GROUPS))
        if bad:
            raise MetadataError(f"unknown age groups: {bad}")

    def quality_report(self) -> dict[str, int]:
        """Counts of missing / placeholder labels (does not raise)."""
        return {
            "rows": len(self.df),
            "missing_values": int(self.df[list(REQUIRED_COLUMNS)].isna().sum().sum()),
            "unknown_gender": int((self.df["gender"] == "Unknown").sum()),
        }

    def distribution(self, column: str) -> pd.Series:
        return self.df[column].value_counts()

    def crosstab(self, rows: str = "ethnicity", cols: str = "age_group") -> pd.DataFrame:
        table = pd.crosstab(self.df[rows], self.df[cols])
        if cols == "age_group":
            table = table.reindex(columns=[g for g in AGE_GROUPS if g in table.columns])
        return table
