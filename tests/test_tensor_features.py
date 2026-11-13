import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdconn.tensor_features import invariants, edge_features


def rot(axis_angle):
    a = np.asarray(axis_angle, float); t = np.linalg.norm(a)
    if t == 0: return np.eye(3)
    k = a / t; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def tensor(l1, l2, l3, R=np.eye(3)):
    return R @ np.diag([l1, l2, l3]) @ R.T


def test_invariants_known_values():
    inv = invariants(tensor(1.7e-3, 0.3e-3, 0.3e-3))
    assert np.isclose(inv["md"], 0.7667e-3, rtol=1e-3)
    assert np.isclose(inv["ad"], 1.7e-3) and np.isclose(inv["rd"], 0.3e-3)
    # axial formula: FA = |l1-l2| / sqrt(l1^2 + 2 l2^2)
    assert np.isclose(inv["fa"], 1.4 / np.sqrt(1.7**2 + 2 * 0.3**2))
    assert np.isclose(invariants(np.eye(3))["fa"], 0)


def test_md_identity():
    rng = np.random.default_rng(0)
    for _ in range(50):
        l = np.sort(rng.uniform(0.1, 2, 3))[::-1]
        inv = invariants(tensor(*l, rot(rng.normal(size=3))))
        assert np.isclose(inv["md"], (inv["ad"] + 2 * inv["rd"]) / 3)


def test_identical_samples_have_no_defect():
    T = np.repeat(tensor(1.5, 0.4, 0.3, rot([0.3, 0.2, 0.1]))[None], 20, axis=0)
    f = edge_features(T)
    for m in ["fa", "md", "ad", "rd"]:
        assert abs(f[f"defect_{m}"]) < 1e-12 and f[f"sd_{m}"] < 1e-12
    assert f["dispersion"] < 1e-12 and abs(f["q_loss"]) < 1e-12


def test_md_defect_is_zero_by_linearity():
    rng = np.random.default_rng(1)
    T = np.array([tensor(*np.sort(rng.uniform(0.2, 2, 3))[::-1], rot(rng.normal(size=3))) for _ in range(100)])
    assert abs(edge_features(T, rng.uniform(0.1, 1, 100))["defect_md"]) < 1e-12


def test_orientation_spread_lowers_fa_of_mean():
    # same shape, two orthogonal orientations: mean tensor is less anisotropic
    a = tensor(1.7, 0.3, 0.3)
    b = tensor(1.7, 0.3, 0.3, rot([0, 0, np.pi / 2]))
    f = edge_features(np.array([a, b]))
    assert f["sd_fa"] < 1e-12            # every sample has the same FA
    assert f["defect_fa"] > 0.3           # but FA of the mean is much lower
    assert np.isclose(f["dispersion"], 0.5)
    assert 0 < f["q_loss"] < 1
