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
    S = [t[None] for t in T]  # 100 one-sample streamlines with weights
    assert abs(edge_features(S, rng.uniform(0.1, 1, 100))["defect_md"]) < 1e-12


def test_orientation_spread_lowers_fa_of_mean():
    # same shape, two orthogonal orientations: mean tensor is less anisotropic
    a = tensor(1.7, 0.3, 0.3)
    b = tensor(1.7, 0.3, 0.3, rot([0, 0, np.pi / 2]))
    f = edge_features(np.array([a, b]))
    assert f["sd_fa"] < 1e-12            # every sample has the same FA
    assert f["defect_fa"] > 0.3           # but FA of the mean is much lower
    assert np.isclose(f["dispersion"], 0.5)
    assert 0 < f["q_loss"] < 1


def test_westin_shapes_sum_to_one():
    rng = np.random.default_rng(3)
    for _ in range(20):
        l = np.sort(rng.uniform(0.1, 2, 3))[::-1]
        i = invariants(tensor(*l, rot(rng.normal(size=3))))
        assert np.isclose(i["cl"] + i["cp"] + i["cs"], 1)


def test_variance_decomposition_and_streamline_weighting():
    rng = np.random.default_rng(4)
    S = [np.array([tensor(*np.sort(rng.uniform(0.2, 2, 3))[::-1]) for _ in range(n)]) for n in (5, 12, 30)]
    wk = np.array([0.2, 0.5, 0.3])
    f = edge_features(S, wk)
    for m in ["fa", "md", "cp"]:
        assert np.isclose(f[f"sd_{m}"] ** 2, f[f"sd_within_{m}"] ** 2 + f[f"sd_between_{m}"] ** 2)
    # mean = streamline means averaged with streamline weights (tck2connectome convention)
    sm = np.array([invariants(s)["fa"].mean() for s in S])
    assert np.isclose(f["mean_fa"], wk @ sm)
    assert f["n_streamlines"] == 3 and f["n_samples"] == 47


def test_alignment_with_tangent():
    pts = np.c_[np.linspace(0, 10, 20), np.zeros(20), np.zeros(20)]   # straight line along x
    along = np.repeat(tensor(1.7, 0.3, 0.3)[None], 20, axis=0)          # fibre along x
    across = np.repeat(tensor(1.7, 0.3, 0.3, rot([0, 0, np.pi / 2]))[None], 20, axis=0)
    assert np.isclose(edge_features([along], points=[pts])["align_mean"], 1)
    assert edge_features([across], points=[pts])["align_mean"] < 1e-12


def test_curvature_alone_creates_defect_but_keeps_alignment():
    # tensors follow a curved tract perfectly: FA defect and dispersion appear,
    # alignment with the tangent stays 1 -> geometry, not microstructure
    th = np.linspace(0, np.pi / 2, 40)
    pts = np.c_[np.cos(th), np.sin(th), np.zeros_like(th)] * 10
    T = []
    for t in tangents_of(pts):
        a = np.cross([1, 0, 0], t); ang = np.arccos(np.clip(t[0], -1, 1))
        T.append(tensor(1.7, 0.3, 0.3, rot(a / np.linalg.norm(a) * ang if np.linalg.norm(a) > 0 else [0, 0, 0])))
    f = edge_features([np.array(T)], points=[pts])
    assert f["defect_fa"] > 0.05 and f["dispersion"] > 0.05
    assert f["align_mean"] > 0.999


def tangents_of(p):
    from pdconn.tensor_features import tangents
    return tangents(p)
