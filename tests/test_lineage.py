from pathlib import Path

import pandas as pd
import pytest

from facedata.dataset.lineage import LineageError, build_lineage, image_number

META = Path(__file__).resolve().parents[1] / "data" / "metadata"


def synthetic_tables(n=40, seed=42):
    consolidated = pd.DataFrame({
        "id": range(1, n + 1),
        "filename": [f"img{i:03d}.png" for i in range(1, n + 1)],
        "ethnicity": ["A", "B"] * (n // 2),
        "age_group": ["13-17", "41plus", "21-25", "26-40"] * (n // 4),
        "gender": ["male", "female"] * (n // 2),
    })
    shuffled = consolidated.sample(frac=1, random_state=seed).reset_index(drop=True)
    restored = shuffled.drop(columns="id").assign(filename=[f"img{i+1:03d}.png" for i in range(n)])
    final = restored.assign(filename=[f"img{i+1:04d}.png" for i in range(n)])
    return consolidated, restored, final


def test_lineage_roundtrip_on_synthetic_tables():
    cons, rest, fin = synthetic_tables()
    lin = build_lineage(cons, rest, fin)
    assert lin["final_filename"].iloc[0] == "img0001.png"
    # every consolidated image appears exactly once
    assert sorted(lin["consolidated_filename"]) == sorted(cons["filename"])
    # labels follow the image back to its source row
    merged = lin.merge(cons, left_on="consolidated_filename", right_on="filename", suffixes=("", "_src"))
    assert (merged["age_group"] == merged["age_group_src"]).all()


def test_wrong_order_is_detected():
    cons, rest, fin = synthetic_tables()
    with pytest.raises(LineageError):
        build_lineage(cons, rest.iloc[::-1].reset_index(drop=True), fin)


def test_image_number():
    assert image_number("img0007.png") == 7 and image_number("img007.png") == 7


@pytest.mark.skipif(not (META / "lineage.csv").exists(), reason="lineage.csv not built")
def test_stored_lineage_is_complete_and_consistent():
    lin = pd.read_csv(META / "lineage.csv")
    assert len(lin) == 1378
    assert lin["final_filename"].is_unique and lin["consolidated_filename"].is_unique
    assert lin["split"].isin(["train", "test"]).all()
    assert int(lin["gender_corrected"].sum()) == 11
