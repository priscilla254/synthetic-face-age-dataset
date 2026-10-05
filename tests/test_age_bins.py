import pytest

from facedata.dataset.age_bins import age_to_group, group_index, normalise_group


@pytest.mark.parametrize("age,expected", [
    (5, "0-12"), (12, "0-12"), (13, "13-17"), (17, "13-17"), (18, "18-20"), (20, "18-20"),
    (21, "21-25"), (25, "21-25"), (26, "26-40"), (40, "26-40"), (41, "41plus"), (80, "41plus"),
])
def test_boundaries(age, expected):
    assert age_to_group(age) == expected


def test_missing_age_is_unknown():
    assert age_to_group(None) == "unknown"
    assert age_to_group(float("nan")) == "unknown"


def test_label_normalisation_and_order():
    assert normalise_group("41+") == "41plus"
    assert group_index("13-17") == 0 and group_index("41+") == 4
    assert group_index("0-12") is None
