import pandas as pd
import pytest

from facedata.dataset.metadata import MetadataError, MetadataStore


def make(**overrides):
    df = pd.DataFrame({"filename": ["a.png", "b.png"], "ethnicity": ["Arabic", "Chinese"],
                       "age_group": ["13-17", "41plus"], "gender": ["male", "female"]})
    for k, v in overrides.items():
        df[k] = v
    return MetadataStore(df)


def test_valid_table_passes():
    make().validate()


def test_missing_column_rejected():
    with pytest.raises(MetadataError):
        MetadataStore(make().df.drop(columns=["gender"])).validate()


def test_duplicate_filenames_rejected():
    with pytest.raises(MetadataError):
        make(filename=["a.png", "a.png"]).validate()


def test_unknown_age_group_rejected():
    with pytest.raises(MetadataError):
        make(age_group=["13-17", "99-100"]).validate()


def test_crosstab_orders_age_groups():
    cols = list(make().crosstab().columns)
    assert cols == ["13-17", "41plus"]


def test_quality_report_counts_unknown_gender():
    assert make(gender=["Unknown", "male"]).quality_report()["unknown_gender"] == 1
