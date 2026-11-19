"""Load MRtrix tck2connectome output as used in the KTH PPMI pipeline.

Expected layout (one folder per subject):
    <root>/<subject>/connectivity/connectome{,_fa,_md,_ad,_rd}.csv       (unfiltered)
    <root>/<subject>/connectivity_sift/connectome{,_fa,_md,_ad,_rd}.csv  (SIFT2)
Files are space-separated, square, upper triangle only (diagonal zero).
"""
from pathlib import Path
import numpy as np

METRICS = {"sc": "", "fa": "_fa", "md": "_md", "ad": "_ad", "rd": "_rd"}
FILTERING = {"raw": "connectivity", "sift": "connectivity_sift"}


def path(root, subject, metric="sc", filtering="sift"):
    return Path(root) / subject / FILTERING[filtering] / f"connectome{METRICS[metric]}.csv"


def load_matrix(root, subject, metric="sc", filtering="sift"):
    """Return the symmetric matrix (zero diagonal) as stored, all nodes kept."""
    W = np.loadtxt(path(root, subject, metric, filtering))
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"not a square matrix: {W.shape}")
    U = np.triu(W, 1)
    if not np.allclose(np.tril(W, -1), 0):  # file already symmetric
        U = np.triu((W + W.T) / 2, 1)
    return U + U.T


def nonempty_nodes(W):
    """Indices of nodes with at least one nonzero connection.

    The two example subjects have 170 nodes, consistent with AAL3 (labels 35,
    36, 81, 82 are empty there); the atlas is not confirmed. A few small
    nuclei may also be missed per subject.
    """
    return np.where(W.sum(1) > 0)[0]


def upper(W):
    """Vector of strictly-upper-triangular entries."""
    return W[np.triu_indices(W.shape[0], 1)]
