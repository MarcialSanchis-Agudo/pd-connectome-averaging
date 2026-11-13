"""Are 'fewer large cliques' and 'more Betti cavities' one effect?

Takes each subject's network at a fixed edge density, applies an increasing
number of degree-preserving edge swaps, and records maximal-clique statistics
and Betti numbers b0..b3 of the flag complex. Also Erdos-Renyi graphs with the
same number of nodes and edges.

Usage:
  python scripts/tda_randomisation.py --root DATA_ROOT --subjects 3104 3105 \
      --metrics sc rd md --density 0.3 --keep high --out results/tda_randomisation.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, networkx as nx
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdconn.io import load_matrix, nonempty_nodes
from pdconn.graphs import threshold, betti_flag, maximal_clique_sizes, rewire

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--subjects", nargs="+", required=True)
p.add_argument("--metrics", nargs="+", default=["sc", "rd", "md"])
p.add_argument("--filtering", default=None, help="raw|sift; default: sift for sc, raw for DTI metrics")
p.add_argument("--density", type=float, default=0.3)
p.add_argument("--keep", default="high", choices=["high", "low"])
p.add_argument("--fractions", type=float, nargs="+", default=[0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 2.0])
p.add_argument("--reps", type=int, default=4)
p.add_argument("--out", default="results/tda_randomisation.json")
a = p.parse_args()


def record(G, n, **meta):
    s = maximal_clique_sizes(G)
    b = betti_flag(G)
    return dict(**meta, n_maximal=len(s), mean_size=float(s.mean()), max_size=int(s.max()),
                b0=b[0], b1=b[1], b2=b[2], b3=b[3])


rows = []
for subj in a.subjects:
    for m in a.metrics:
        filt = a.filtering or ("sift" if m == "sc" else "raw")
        W = load_matrix(a.root, subj, m, filt)
        keep = nonempty_nodes(W)
        W = W[np.ix_(keep, keep)]
        n = W.shape[0]
        G0 = threshold(W, a.density, keep=a.keep if m != "sc" else "high")
        for f in a.fractions:
            for rep in range(a.reps if f > 0 else 1):
                r = record(rewire(G0, f, seed=1000 * rep + int(100 * f)), n,
                           subject=subj, metric=m, filtering=filt, swap_fraction=f, rep=rep)
                rows.append(r); print(r, flush=True)
        for rep in range(a.reps):
            r = record(nx.gnm_random_graph(n, G0.number_of_edges(), seed=rep), n,
                       subject=subj, metric=m, filtering=filt, swap_fraction="ER", rep=rep)
            rows.append(r); print(r, flush=True)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(dict(args=vars(a), rows=rows), open(a.out, "w"), indent=1)

# summary: Spearman between mean maximal clique size and b2, b3 along the rewiring series
import pandas as pd
d = pd.DataFrame(rows)
d = d[d.swap_fraction.astype(str) != "ER"]
print("\nSpearman(mean maximal clique size, Betti) along the rewiring series")
for (s, m), x in d.groupby(["subject", "metric"]):
    print(f"  {s} {m}: b2 {x.mean_size.corr(x.b2, method='spearman'):+.2f}   b3 {x.mean_size.corr(x.b3, method='spearman'):+.2f}")
