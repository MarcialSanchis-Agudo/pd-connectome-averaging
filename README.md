# What averaging hides in structural connectomes of Parkinson's disease

Shared code for Marcial Sanchis Agudo and Ilaria Carannante (KTH).

**Question.** Structural connectomes store one number per pair of regions: a
streamline count, or FA/MD/AD/RD averaged along and across all streamlines.
On PPMI, these mean-based connectomes do not separate PD from healthy controls
(Sinzinger et al., *Front. Hum. Neurosci.* 20:1769103, 2026). We test whether the
PD signal sits in what the averaging removes: the spread of tensor values along
and across streamlines, the gap between "invariant of the mean tensor" and
"mean of the invariant", and orientation dispersion.

The diffusion tensor is a covariance of water displacement and splits into an
isotropic part (MD) and a deviatoric part (anisotropy). Averaging tensors that
point in different directions cancels part of the deviatoric part, as filtering
does in turbulence. That cancelled part is what we measure.

**No subject data is stored here** (PPMI data use agreement). Data stay on
KTH OneDrive / local disks; scripts take a `--root` path.

## Layout

```
src/pdconn/io.py               load tck2connectome CSVs (unfiltered and SIFT2)
src/pdconn/graphs.py           thresholding, flag-complex Betti numbers, rewiring
src/pdconn/tensor_features.py  edge features beyond the mean (needs per-sample tensors)
scripts/check_redundancy.py    how many weightings are independent
scripts/tda_randomisation.py   cliques vs cavities under degree-preserving rewiring
scripts/make_figures.py        figures for the TDA check (keep=high)
scripts/fa_gap_sources.py      what makes the FA gap positive (synthetic)
tests/                         unit tests (pytest), synthetic inputs only
results/figures/               figures from the first two subjects
```

Expected data layout under `--root`:
`<subject>/connectivity{,_sift}/connectome{,_fa,_md,_ad,_rd}.csv`
(space-separated, upper triangle, as written by MRtrix `tck2connectome`).

## Setup

```
pip install -r requirements.txt
pytest -q
python scripts/check_redundancy.py --root DATA --subjects 3104 3105
python scripts/tda_randomisation.py --root DATA --subjects 3104 3105
python scripts/make_figures.py --root DATA
```

## First observations (subjects 3104 and 3105 only, groups unknown)

These are statements about the data structure, not about PD.

1. **MD is redundant (as expected from linearity).** On every edge, MD = (AD + 2 RD)/3 to median relative
   error 3e-4 (99th percentile 0.7-1.3 %). The five weightings carry at most
   four independent families (count + three tensor quantities).
2. **FA is not a function of mean AD and RD, but the gap is mostly tensor
   shape, not averaging.** Streamline-averaged FA exceeds the FA computed from
   averaged AD and RD with the axially symmetric formula by +0.027 to +0.031
   on average (sd ~0.018). `scripts/fa_gap_sources.py` shows that
   non-axial tensors (l2 != l3, e.g. crossing fibres) make this gap positive,
   while heterogeneity of axially symmetric tensors along the tract makes it
   negative. A positive average gap therefore points to tensor shape (planarity),
   not to information revealed by averaging. Testing the averaging hypothesis
   needs per-sample tensors.
3. **Filtering barely reorders edges.** log streamline counts, unfiltered vs
   SIFT2, correlate at r = 0.990 within a subject.
4. **TDA: whether cliques and cavities move together depends on the graph.**
   At edge density 0.3, degree-preserving rewiring always shrinks maximal
   cliques. For streamline-count graphs it also raises b3:
   Spearman(mean maximal clique size, b3) = -0.87 / -0.74 (3104 / 3105).
   For RD and MD the result depends on which edges are kept:
   - keeping the highest values (`--keep high`): RD b3 -0.82 / -0.73,
     MD +0.14 / -0.65;
   - keeping the lowest values (`--keep low`, low diffusivity = better
     integrity): RD b3 +0.42 / +0.51, MD -0.42 / -0.34, and b2 *falls* with
     rewiring (RD b2 +0.93 / +0.73).
   b2 is not consistently coupled in any setting. So "fewer large cliques" and
   "more cavities" are one effect only for streamline counts (and high-RD
   graphs); elsewhere they are distinct. Erdos-Renyi graphs always have small
   cliques (max 7) and many cavities (b2 ~350, b3 ~700-900), so they are a weak
   reference. Open question for the TDA work: is the PD-HC difference a shift
   toward the degree-preserving null? The poster's thresholding direction for
   RD/MD still needs to be matched. Note: we use maximal cliques; the poster may
   count all cliques.

![](results/figures/fig1_cliques_vs_cavities.png)
![](results/figures/fig2_clique_size_distributions.png)

## Next step: edge features beyond the mean

`tensor_features.edge_features(tensors, weights)` takes all tensor samples for
one edge (shape (N, 3, 3); weights e.g. SIFT2 x step length) and returns:

| feature | meaning |
|---|---|
| `mean_<m>` | what current connectomes store (m = fa, md, ad, rd) |
| `sd_<m>` | heterogeneity along/across streamlines |
| `defect_<m>` | mean of m minus m of the mean tensor (zero for MD by linearity) |
| `sd_within_<m>`, `sd_between_<m>` | spread along streamlines vs between streamline means |
| `dispersion` | 1 - largest eigenvalue of the mean of v v^T over principal directions |
| `q_loss` | fraction of deviatoric magnitude cancelled by averaging |
| `align_mean`, `align_sd` | \|cos\| between tensor direction and streamline tangent (needs coordinates) |
| `cl`, `cp`, `cs` | Westin linear / planar / spherical shape (mean, spread, defect) |

Interpretation traps, built into the tests:
- `defect_fa` and `dispersion` grow with ordinary tract curvature (anatomy) even
  when microstructure is perfect; `align_*` separates the two.
- Spread estimates depend on streamline and sample counts; use `n_streamlines`
  and `n_samples` as covariates in group comparisons.

Input needed per subject: the tractogram (.tck), SIFT2 weights, the tensor
image (dt.mif, or its six components) and the parcellation in subject space.
From these we sample tensors along each streamline (`tcksample`) and keep the
coordinates for tangents. Scalar FA/MD/AD/RD samples alone allow spread
features but not the tensor defect or alignment.

## Prior work to position against

Along-tract profiling (tractometry; e.g. AFQ) already looks beyond the
tract-wide mean, including in PD ("White matter differences in Parkinson's
disease mapped using tractometry", bioRxiv 209858 / NeuroImage: Clinical 2017).
NODDI's orientation dispersion index and fixel-based analysis also measure
dispersion. Our contribution has to be specific: beyond-mean features at the
level of the whole connectome, in the same cohort and statistics where mean
features are null, with geometry (curvature) separated from microstructure.

Plan: pilot (~20 labelled subjects) to fix definitions and the analysis
(pre-registered), then the full cohort with the same statistics and
classifiers as the Frontiers paper (TFNBS, BY correction, SVM/GNN), mean vs
beyond-mean features.

## Not done yet / known limitations

- **No sampling pipeline yet.** Nothing here reads .tck files. Planned route
  (MRtrix3, untested on her data): `tck2connectome -out_assignments` for edge
  membership, then `tcksample` on the six tensor components (or FA/MD/AD/RD)
  for a random subset of streamlines per edge (e.g. at most 200), saving
  coordinates for tangents. Sampling every point of every streamline is not
  feasible: about 10^7 streamlines x 100 points x 6 values is tens of GB per
  subject.
- **Partial volume at the ends.** Streamline ends lie in grey matter or near
  CSF; `edge_features(..., trim=k)` drops k samples at each end. The right k
  has to be chosen on the pilot (before looking at labels).
- **Noise.** PPMI is single-shell b = 1000, 64 directions. Spread along a
  streamline may be dominated by noise and interpolation. The pilot must check
  this first (e.g. spread vs SNR, test-retest if available) before any group
  comparison.
- **Free water.** Free-water imaging is an established PD marker (e.g. in the
  substantia nigra) and is the isotropic part of the same tensor split; any
  isotropic feature here must be compared against it.
- **TDA figures** use `--keep high` for RD/MD; results differ with `--keep low`.

## Status of claims (13 Nov)

| claim | status |
|---|---|
| MD = (AD + 2 RD)/3 per edge | reproduced by `check_redundancy.py` (median 3e-4) |
| FA gap +0.03 | reproduced; reinterpreted as tensor shape, not averaging |
| log SC raw vs SIFT2 r = 0.99 | reproduced |
| TDA coupling | holds for SC; depends on threshold direction for RD/MD |
| PD-related anything | not tested (no labels) |

## Conventions

Every number in a manuscript comes from a script in this repository and is
reproducible from it. Negative results are reported.
