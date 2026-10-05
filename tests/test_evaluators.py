import pandas as pd

from facedata.evaluation.age_eval import AgeEstimationEvaluator
from facedata.evaluation.fairface_eval import EXTENDED_MAPPING, NOTEBOOK_MAPPING, FairFaceEvaluator
from facedata.evaluation.human_eval import HumanEvalAnalyser


def test_age_evaluator_exact_and_adjacent_accuracy():
    gt = pd.DataFrame({"filename": [f"img{i}.png" for i in range(4)],
                       "age_group": ["13-17", "18-20", "26-40", "41+"]})
    preds = pd.DataFrame({"filename": ["img0_00.png", "img1_00.png", "img2_00.png", "img3_00.png"],
                          "age": [15, 24, 45, 60]})
    preds["filename"] = preds["filename"].str.replace(r"_\d+(?=\.\w+$)", "", regex=True)
    from facedata.dataset.age_bins import age_to_group
    preds["predicted_group"] = preds["age"].map(age_to_group)
    res = AgeEstimationEvaluator().evaluate(preds, gt)
    # 15->13-17 and 60->41plus are exact; 24 (true 18-20) and 45 (true 26-40) are one bin too old
    assert res.accuracy == 0.5
    assert res.adjacent_accuracy == 1.0
    assert res.per_group.loc["18-20", "mean_signed_bin_error"] == 1
    assert res.per_group.loc["13-17", "mean_signed_bin_error"] == 0


def test_fairface_unmapped_rows_are_reported_not_hidden():
    gt = pd.DataFrame({"filename": ["a.png", "b.png"], "ethnicity": ["East African", "Chinese"]})
    pred = pd.DataFrame({"filename": ["a.png", "b.png"], "race": ["Black", "East Asian"]})
    strict = FairFaceEvaluator(NOTEBOOK_MAPPING).evaluate(pred, gt)
    full = FairFaceEvaluator(EXTENDED_MAPPING).evaluate(pred, gt)
    assert strict.n_unmapped_dropped == 1 and strict.n_rows == 1
    assert full.n_unmapped_dropped == 0 and full.accuracy == 1.0


def test_fairface_filename_extraction(tmp_path):
    p = tmp_path / "p.csv"
    pd.DataFrame({"face_name_align": ["detected_faces/img0752_face0.png"], "race": ["Black"]}).to_csv(p, index=False)
    assert FairFaceEvaluator.load_predictions(p)["filename"].iloc[0] == "img0752.png"


def test_human_summary_has_ci_inside_range():
    df = pd.DataFrame({"filename": list("abcdef"), "ethnicity": ["X"] * 6, "age_group": ["13-17"] * 6,
                       "realism": [1, 2, 3, 4, 5, 3], "age_appropriateness": [3] * 6, "ethnic_consistency": [4] * 6})
    s = HumanEvalAnalyser(df).summary().set_index("criterion")
    r = s.loc["realism"]
    assert r["ci95_low"] <= r["mean"] <= r["ci95_high"]
    assert s.loc["age_appropriateness", "ci95_low"] == 3
