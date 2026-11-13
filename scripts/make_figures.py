"""Figures for the TDA randomisation check.

Usage: python scripts/make_figures.py --root DATA_ROOT --json results/tda_randomisation.json \
          --subject-for-fig2 3104 --outdir results/figures
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdconn.io import load_matrix, nonempty_nodes
from pdconn.graphs import threshold, maximal_clique_sizes, rewire

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--json", default="results/tda_randomisation.json")
p.add_argument("--subject-for-fig2", default=None)
p.add_argument("--outdir", default="results/figures")
a = p.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
J = json.load(open(a.json)); args = J["args"]
d = pd.DataFrame(J["rows"])
er = d[d.swap_fraction.astype(str) == "ER"]
d = d[d.swap_fraction.astype(str) != "ER"].copy(); d["swap_fraction"] = d.swap_fraction.astype(float)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "xtick.color": "#555", "ytick.color": "#555"})
ramp = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6"]
fr = sorted(d.swap_fraction.unique()); cmap = {f: ramp[min(i, len(ramp) - 1)] for i, f in enumerate(fr)}
names = {"sc": "Streamline count (SIFT2)", "rd": "RD-weighted", "md": "MD-weighted", "fa": "FA-weighted", "ad": "AD-weighted"}
markers = ["o", "^", "s", "D", "v", "P"]
subjects = list(dict.fromkeys(d.subject)); metrics = list(dict.fromkeys(d.metric))

fig, ax = plt.subplots(2, len(metrics), figsize=(4.4 * len(metrics), 7.6), squeeze=False)
for j, m in enumerate(metrics):
    for i, b in enumerate(["b2", "b3"]):
        A = ax[i, j]; A.grid(color="#e1e0d9", lw=.6); A.set_axisbelow(True)
        for s, mk in zip(subjects, markers):
            x = d[(d.subject == s) & (d.metric == m)]
            g = x.groupby("swap_fraction")[["mean_size", b]].mean()
            A.plot(g.mean_size, g[b], color="#c3c2b7", lw=1, zorder=1)
            A.scatter(x.mean_size, x[b], c=[cmap[f] for f in x.swap_fraction], marker=mk, s=34, edgecolor="white", lw=.8, zorder=2)
            r = g.loc[0.0]
            A.scatter([r.mean_size], [r[b]], marker=mk, s=110, facecolor="none", edgecolor="#d95926", lw=2, zorder=3)
            A.annotate(f"real {s}", (r.mean_size, r[b]), xytext=(6, 6), textcoords="offset points", fontsize=8.5)
        e = er[er.metric == m]
        A.text(0, 1.01, f"ER random graph (off scale): {b.replace('b', 'β')} ≈ {e[b].mean():.0f}, mean clique size ≈ {e.mean_size.mean():.1f}",
               transform=A.transAxes, fontsize=8, color="#666", va="bottom")
        if i == 0: A.set_title(names.get(m, m) + "\n", loc="left", fontsize=11)
        A.set_ylabel({"b2": "β₂ (2-d cavities)", "b3": "β₃ (3-d cavities)"}[b])
        if i == 1: A.set_xlabel("mean size of maximal cliques")
h = [Line2D([], [], marker="o", ls="", color=cmap[f], markersize=7, label=f"{f:g}×E swaps") for f in fr]
h += [Line2D([], [], marker="o", ls="", mfc="none", mec="#d95926", mew=2, markersize=9, label="real network")]
h += [Line2D([], [], marker=mk, ls="", color="#777", label=f"subject {s}") for s, mk in zip(subjects, markers)]
fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=8.5)
fig.suptitle(f"Degree-preserving randomisation at edge density {args['density']}: cliques vs cavities", x=0.01, ha="left", fontsize=12.5)
fig.tight_layout(rect=(0, 0.08, 1, 0.96)); fig.savefig(out / "fig1_cliques_vs_cavities.png", dpi=160)

s = a.subject_for_fig2 or subjects[0]
fig, ax = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.6), squeeze=False)
for j, m in enumerate(metrics):
    A = ax[0, j]; A.grid(color="#e1e0d9", lw=.6, axis="y"); A.set_axisbelow(True)
    filt = "sift" if m == "sc" else "raw"
    W = load_matrix(a.root, s, m, filt); k = nonempty_nodes(W); W = W[np.ix_(k, k)]
    G0 = threshold(W, args["density"], keep=args["keep"] if m != "sc" else "high")
    for G, lab, c in [(G0, "real", "#d95926"), (rewire(G0, 1.0, seed=7), "rewired (degrees kept)", "#2a78d6"),
                      (nx.gnm_random_graph(W.shape[0], G0.number_of_edges(), seed=7), "ER random", "#9a9a92")]:
        hst = np.bincount(maximal_clique_sizes(G), minlength=32)[:32]
        A.plot(np.arange(32), hst / hst.sum(), color=c, lw=2, label=lab)
    A.set_title(f"{names.get(m, m)}, subject {s}", loc="left", fontsize=10.5)
    A.set_xlabel("maximal clique size (regions)")
    if j == 0: A.set_ylabel("fraction of maximal cliques"); A.legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(out / "fig2_clique_size_distributions.png", dpi=160)
print("wrote", out)
