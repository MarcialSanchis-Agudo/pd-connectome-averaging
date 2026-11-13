"""Edge features beyond the mean: what averaging along/across streamlines removes.

Input for one edge (pair of regions): the diffusion tensors sampled along each
streamline connecting the pair, as a list with one array of shape (n_k, 3, 3)
per streamline, optionally with one weight per streamline (e.g. SIFT2).
Each streamline counts with its weight, spread evenly over its samples, which
matches how tck2connectome builds mean-based connectomes from per-streamline
means.

Conventions: eigenvalues l1 >= l2 >= l3.
    MD = (l1 + l2 + l3)/3,  AD = l1,  RD = (l2 + l3)/2,
    FA = sqrt(3/2) * |lambda - MD| / |lambda|.
    Westin shape: cl = (l1 - l2)/l1, cp = (l2 - l3)/l1, cs = l3/l1 (cl + cp + cs = 1).
D = MD I + Q splits the tensor into isotropic (MD) and deviatoric (Q) parts.

Caveats that matter for interpretation
- defect_fa > 0 does NOT by itself mean heterogeneity. Mean FA vs FA of the
  mean tensor mixes (a) orientation change along the tract, which includes
  ordinary tract curvature (anatomy), (b) shape heterogeneity, which pushes
  the defect the other way for nearly parallel tensors. Use `align_*` (tensor
  direction vs streamline tangent) to separate microstructure from geometry.
- Spread estimates depend on the number of samples and streamlines; compare
  groups with n_streamlines / n_samples as covariates.
"""
import numpy as np

INV = ("md", "ad", "rd", "fa", "cl", "cp", "cs")


def invariants(D):
    """MD, AD, RD, FA and Westin cl, cp, cs for tensors of shape (..., 3, 3)."""
    lam = np.linalg.eigvalsh(np.asarray(D, float))[..., ::-1]
    md = lam.mean(-1)
    dev = lam - md[..., None]
    fa = np.sqrt(1.5) * np.linalg.norm(dev, axis=-1) / np.linalg.norm(lam, axis=-1)
    l1 = lam[..., 0]
    return {"md": md, "ad": l1, "rd": lam[..., 1:].mean(-1), "fa": fa,
            "cl": (lam[..., 0] - lam[..., 1]) / l1, "cp": (lam[..., 1] - lam[..., 2]) / l1,
            "cs": lam[..., 2] / l1}


def principal_direction(D):
    return np.linalg.eigh(np.asarray(D, float))[1][..., :, -1]


def tangents(points):
    """Unit tangents of a polyline (n, 3) by central differences."""
    t = np.gradient(np.asarray(points, float), axis=0)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def _deviatoric_norm(D):
    tr = np.trace(D, axis1=-2, axis2=-1)[..., None, None] / 3
    return np.linalg.norm(D - tr * np.eye(3), axis=(-2, -1))


def edge_features(streamlines, weights=None, points=None):
    """Features for one edge.

    streamlines : list of arrays (n_k, 3, 3), tensors along streamline k.
                  A single (N, 3, 3) array is treated as one streamline.
    weights     : per-streamline weights (default 1).
    points      : optional list of (n_k, 3) coordinates for tangent alignment.

    Returns, for each invariant m in INV:
      mean_m        weighted mean (what mean-based connectomes store; streamline
                    means averaged with streamline weights)
      sd_m          total weighted SD over all samples
      sd_within_m   SD along streamlines (pooled), sd_between_m SD of
                    streamline means; sd_m**2 = sd_within_m**2 + sd_between_m**2
      defect_m      mean_m minus m of the weighted-mean tensor (0 for MD)
    and
      dispersion    1 - top eigenvalue of the weighted mean of v v^T
      q_loss        1 - |Q(mean D)| / mean |Q(D)|
      align_mean, align_sd   |cos| between principal direction and tangent
                    (only if points given)
      n_streamlines, n_samples
    """
    if isinstance(streamlines, np.ndarray) and streamlines.ndim == 3:
        streamlines = [streamlines]
    S = [np.asarray(s, float) for s in streamlines]
    K = len(S)
    wk = np.ones(K) if weights is None else np.asarray(weights, float)
    wk = wk / wk.sum()
    T = np.concatenate(S)
    w = np.concatenate([np.full(len(s), wk[k] / len(s)) for k, s in enumerate(S)])
    sid = np.concatenate([np.full(len(s), k) for k, s in enumerate(S)])

    inv = invariants(T)
    inv_bar = invariants(np.einsum("i,ijk->jk", w, T))
    out = {"n_streamlines": K, "n_samples": len(T)}
    for m in INV:
        x = inv[m]
        mu = float(w @ x)
        smean = np.array([x[sid == k].mean() for k in range(K)])
        between = float(wk @ (smean - mu) ** 2)
        within = float(w @ (x - smean[sid]) ** 2)
        out[f"mean_{m}"] = mu
        out[f"sd_{m}"] = float(np.sqrt(w @ (x - mu) ** 2))
        out[f"sd_within_{m}"] = np.sqrt(within)
        out[f"sd_between_{m}"] = np.sqrt(between)
        out[f"defect_{m}"] = mu - float(inv_bar[m])
    v = principal_direction(T)
    out["dispersion"] = float(1 - np.linalg.eigvalsh(np.einsum("i,ij,ik->jk", w, v, v))[-1])
    out["q_loss"] = float(1 - _deviatoric_norm(np.einsum("i,ijk->jk", w, T)) / (w @ _deviatoric_norm(T)))
    if points is not None:
        tg = np.concatenate([tangents(p) for p in points])
        c = np.abs(np.einsum("ij,ij->i", v, tg))
        out["align_mean"] = float(w @ c)
        out["align_sd"] = float(np.sqrt(w @ (c - w @ c) ** 2))
    return out
