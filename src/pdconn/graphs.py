"""Thresholding and topology of connectome graphs."""
import numpy as np
import networkx as nx
import gudhi


def threshold(W, density, keep="high"):
    """Binary graph keeping a fraction `density` of all node pairs.

    keep="high" keeps the largest weights, keep="low" the smallest nonzero
    ones (e.g. for diffusivities, where low means better integrity).
    Ties are broken by index order (stable sort).
    """
    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    w = W[iu]
    m = int(round(density * len(w)))
    if keep == "high":
        order = np.argsort(-w, kind="stable")
    elif keep == "low":
        w2 = np.where(w > 0, w, np.inf)
        order = np.argsort(w2, kind="stable")
    else:
        raise ValueError(keep)
    idx = order[:m]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(iu[0][idx].tolist(), iu[1][idx].tolist()))
    return G


def betti_flag(G, maxdim=3):
    """Betti numbers b0..b_maxdim of the clique (flag) complex of G.

    Uses gudhi edge collapse (preserves homology) before expansion, which
    keeps dense brain graphs (clique number ~30) tractable.
    """
    st = gudhi.SimplexTree()
    for v in G.nodes:
        st.insert([int(v)])
    for a, b in G.edges:
        st.insert([int(a), int(b)])
    st.collapse_edges(nb_iterations=10)
    st.expansion(maxdim + 1)
    st.compute_persistence(persistence_dim_max=True)
    b = list(st.betti_numbers()) + [0] * (maxdim + 1)
    return b[: maxdim + 1]


def maximal_clique_sizes(G):
    return np.array([len(c) for c in nx.find_cliques(G)])


def rewire(G, fraction, seed=0):
    """Degree-preserving double-edge swaps: fraction x number of edges."""
    H = G.copy()
    k = int(fraction * H.number_of_edges())
    if k > 0:
        nx.double_edge_swap(H, nswap=k, max_tries=100 * k, seed=seed)
    return H
