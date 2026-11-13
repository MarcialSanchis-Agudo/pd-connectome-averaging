import sys
from pathlib import Path
import numpy as np, networkx as nx
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdconn.graphs import threshold, betti_flag, rewire


def test_threshold_density_and_direction():
    rng = np.random.default_rng(0); n = 30
    W = np.triu(rng.uniform(1, 2, (n, n)), 1); W = W + W.T
    G = threshold(W, 0.2)
    assert G.number_of_edges() == round(0.2 * n * (n - 1) / 2)
    w_in = min(W[a, b] for a, b in G.edges)
    Gl = threshold(W, 0.2, keep="low")
    assert max(W[a, b] for a, b in Gl.edges) <= w_in


def test_betti_known_complexes():
    assert betti_flag(nx.cycle_graph(6))[:2] == [1, 1]          # circle
    octa = nx.complete_multipartite_graph(2, 2, 2)              # octahedron = 2-sphere
    assert betti_flag(octa)[:3] == [1, 0, 1]
    c4 = nx.complete_multipartite_graph(2, 2, 2, 2)             # cross-polytope = 3-sphere
    assert betti_flag(c4) == [1, 0, 0, 1]
    assert betti_flag(nx.complete_graph(8)) == [1, 0, 0, 0]


def test_rewire_keeps_degrees():
    G = nx.gnm_random_graph(40, 200, seed=1)
    H = rewire(G, 1.0, seed=2)
    assert dict(G.degree) == dict(H.degree) and H.number_of_edges() == 200
