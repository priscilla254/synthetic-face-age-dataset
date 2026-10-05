import numpy as np

from facedata.evaluation.fid_eval import FIDEvaluator, feature_statistics, frechet_distance


def test_identical_distributions_have_zero_distance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 8))
    mu, s = feature_statistics(x)
    assert abs(frechet_distance(mu, s, mu, s)) < 1e-6


def test_mean_shift_equals_squared_distance_when_covariances_match():
    s = np.eye(4)
    d = frechet_distance(np.zeros(4), s, np.array([3.0, 0, 0, 0]), s)
    assert abs(d - 9.0) < 1e-9


def test_scaled_covariance_matches_closed_form():
    # S2 = 4*I -> Tr(I + 4I - 2*sqrt(4I)) = 4*(1+4-4) = 4 for d=4
    d = frechet_distance(np.zeros(4), np.eye(4), np.zeros(4), 4 * np.eye(4))
    assert abs(d - 4.0) < 1e-9


def test_evaluator_uses_extractor():
    rng = np.random.default_rng(1)
    vectors = {i: rng.normal(size=6) for i in range(40)}
    ev = FIDEvaluator(lambda p: vectors[int(p.name)])
    same = ev.score([__import__("pathlib").Path(str(i)) for i in range(40)],
                    [__import__("pathlib").Path(str(i)) for i in range(40)])
    assert abs(same) < 1e-6
