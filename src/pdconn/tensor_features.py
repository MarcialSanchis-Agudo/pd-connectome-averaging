"""Edge features beyond the mean: what averaging along/across streamlines removes.

Input for one edge (pair of regions): an array of diffusion tensors sampled
along all streamlines connecting the pair, shape (N, 3, 3), optionally with
weights (e.g. SIFT2 weights times step length), shape (N,).

Conventions: eigenvalues l1 >= l2 >= l3.
    MD = (l1 + l2 + l3)/3,  AD = l1,  RD = (l2 + l3)/2,
    FA = sqrt(3/2) * |D - MD I| / |D|   (Frobenius norms).
D = MD I + Q splits the tensor into isotropic (MD) and deviatoric (Q) parts.
"""
import numpy as np


def _w(n, weights):
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    return w / w.sum()


def invariants(D):
    """MD, AD, RD, FA for tensors of shape (..., 3, 3)."""
    D = np.asarray(D, float)
    lam = np.linalg.eigvalsh(D)[..., ::-1]
    md = lam.mean(-1)
    ad = lam[..., 0]
    rd = lam[..., 1:].mean(-1)
    dev = lam - md[..., None]
    fa = np.sqrt(1.5) * np.linalg.norm(dev, axis=-1) / np.linalg.norm(lam, axis=-1)
    return {"md": md, "ad": ad, "rd": rd, "fa": fa}


def principal_direction(D):
    return np.linalg.eigh(np.asarray(D, float))[1][..., :, -1]


def edge_features(tensors, weights=None):
    """Mean-based and beyond-mean features for one edge.

    Returns a dict with
      mean_<m>      : weighted mean of the invariant m over samples (what the
                      current connectomes store);
      sd_<m>        : weighted standard deviation of m (heterogeneity);
      defect_<m>    : mean_<m> minus m of the weighted-mean tensor
                      (zero for MD by linearity; for FA it combines
                      heterogeneity and orientation spread);
      dispersion    : 1 - largest eigenvalue of the weighted mean of v v^T
                      over principal directions v (0 = all aligned,
                      2/3 = isotropic spread);
      q_loss        : 1 - |Q(mean D)| / mean |Q(D)|, fraction of the
                      deviatoric magnitude cancelled by averaging.
    """
    T = np.asarray(tensors, float)
    w = _w(len(T), weights)
    inv = invariants(T)
    Dbar = np.einsum("i,ijk->jk", w, T)
    inv_bar = invariants(Dbar)
    out = {}
    for m, x in inv.items():
        mu = float(w @ x)
        out[f"mean_{m}"] = mu
        out[f"sd_{m}"] = float(np.sqrt(w @ (x - mu) ** 2))
        out[f"defect_{m}"] = mu - float(inv_bar[m])
    v = principal_direction(T)
    S = np.einsum("i,ij,ik->jk", w, v, v)
    out["dispersion"] = float(1 - np.linalg.eigvalsh(S)[-1])
    I = np.eye(3)
    q = lambda D: np.linalg.norm(D - np.trace(D, axis1=-2, axis2=-1)[..., None, None] / 3 * I, axis=(-2, -1))
    out["q_loss"] = float(1 - q(Dbar) / (w @ q(T)))
    return out
