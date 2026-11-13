"""Check how much independent information the five connectome weightings carry.

Usage: python scripts/check_redundancy.py --root DATA_ROOT --subjects 3104 3105

Reports, per subject and filtering:
  1. MD vs (AD + 2 RD)/3 per edge (an exact identity at tensor level;
     holds per edge because streamline averaging is linear).
  2. FA_mean - FA(AD_mean, RD_mean) with the axially symmetric formula
     (nonlinear: a mix of along-tract heterogeneity and l2 != l3).
  3. Correlation of log streamline counts, unfiltered vs SIFT2.
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdconn.io import load_matrix, upper

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--subjects", nargs="+", required=True)
a = p.parse_args()

for s in a.subjects:
    for filt in ["raw", "sift"]:
        M = {m: upper(load_matrix(a.root, s, m, filt)) for m in ["sc", "fa", "md", "ad", "rd"]}
        k = M["md"] > 0
        rel = np.abs(M["md"][k] - (M["ad"][k] + 2 * M["rd"][k]) / 3) / M["md"][k]
        ad, rd = M["ad"][k], M["rd"][k]
        fa_ax = np.abs(ad - rd) / np.sqrt(ad**2 + 2 * rd**2)
        gap = M["fa"][k] - fa_ax
        print(f"{s} {filt:4s}  MD identity rel.err median {np.median(rel):.1e} p99 {np.percentile(rel, 99):.1e}"
              f" | FA gap mean {gap.mean():+.4f} sd {gap.std():.4f} | edges {k.sum()}")
    sc = [upper(load_matrix(a.root, s, "sc", f)) for f in ["raw", "sift"]]
    k = (sc[0] > 0) & (sc[1] > 0)
    print(f"{s}       corr(log SC raw, log SC SIFT2) = {np.corrcoef(np.log(sc[0][k]), np.log(sc[1][k]))[0, 1]:.4f}")
