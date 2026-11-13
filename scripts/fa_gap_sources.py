"""What can make FA_mean - FA_axial(AD_mean, RD_mean) positive?

FA_axial(AD, RD) = |AD - RD| / sqrt(AD^2 + 2 RD^2) is exact only for
axially symmetric tensors (l2 = l3). Two effects enter the gap measured on the
FA/AD/RD-weighted connectomes:
  (a) non-axial tensors (l2 != l3, planar shapes, e.g. crossing fibres): gap > 0
  (b) heterogeneity of axially symmetric tensors along the tract: gap < 0 here
So a positive average gap (+0.03 in subjects 3104/3105) points to (a), a
per-voxel shape property, not to heterogeneity revealed by averaging.
"""
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdconn.tensor_features import invariants

fa_ax = lambda ad, rd: np.abs(ad - rd) / np.sqrt(ad**2 + 2 * rd**2)
print("(a) single non-axial tensor, no averaging")
for l in [(1.5, 0.4, 0.4), (1.5, 0.5, 0.3), (1.5, 0.6, 0.2), (1.2, 0.7, 0.4)]:
    i = invariants(np.diag(l))
    print(f"   eig {l}: FA {i['fa']:.3f}  FA_axial {fa_ax(i['ad'], i['rd']):.3f}  gap {i['fa'] - fa_ax(i['ad'], i['rd']):+.3f}")
print("(b) axially symmetric tensors, lognormal spread of l1 and l2=l3")
rng = np.random.default_rng(0)
for s in [0.05, 0.15, 0.3]:
    l1 = 1.4 * np.exp(rng.normal(0, s, 20000)); l2 = 0.4 * np.exp(rng.normal(0, s, 20000))
    fa = fa_ax(l1, l2)
    print(f"   sd {s:.2f}: mean FA {fa.mean():.3f}  FA_axial(means) {fa_ax(l1.mean(), l2.mean()):.3f}  gap {fa.mean() - fa_ax(l1.mean(), l2.mean()):+.3f}")
