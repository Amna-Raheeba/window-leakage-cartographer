# Window Leakage Cartographer

## 1. What this is

Slice a run-to-failure trajectory into sliding windows, shuffle the
windows, and hold out 25% as a test set — the standard recipe in
time-series ML, and a leaky one. A window that starts five timesteps
after a training window shares most of its raw data with it, so the
"test" score partly measures how well the model recognises data it
has effectively already seen.

This tool *maps* that leakage and *prices* it. It builds the
window-overlap graph of a split, finds the connected components that
straddle the split, reports leakage mass per test window, and — by
removal replay with a simple predictor — attributes the score
inflation to each leakage component individually, not just as one
total. Everything is offline, deterministic, and pure Python stdlib:
seeded simulated fleets with planted ground truth, so every number
below is measured, not asserted.

## 2. The novel mechanism

1. **Window-overlap graph.** Nodes are windows. Two windows from the
   same unit are joined when their raw timestep ranges overlap
   (weight = shared timesteps / window length) or when their ranges
   are disjoint but their raw values match within a tolerance band at
   ≥85% of positions (the near-duplicate channel, for leakage that
   survives re-windowing). Connected components that contain both
   train and test windows are the leakage components; a test window's
   leakage mass combines its edge weights to train windows as
   1 − Π(1 − w).
2. **Four scores on the same test windows** (kNN RUL regressor,
   k = 3): the *leaky* score (full training set); the *purged* score
   (every train window that is a direct leakage source is removed —
   purged cross-validation driven by the map); the
   *leave-own-unit-out* (LOUO) score, where each test window is
   scored by a model that never saw its unit — the ground-truth
   honest score for these exact windows; and a *unit-disjoint split*
   reference.
3. **Per-component removal replay.** For each straddling component,
   remove only that component's leaked train windows and re-score
   its test windows. Summed over components, the replay gains are
   the inflation the map itself explains; divided by the total LOUO
   optimism they give the recovery share. The remainder is the
   *same-unit adjacency premium* — windows that share no timesteps
   still resemble their own trajectory — which no overlap audit can
   remove and only a unit-disjoint split prices.

## 3. Prior art and the differentiator

Group leakage and dataset contamination are well documented for text
(n-gram overlap, MinHash dedup, GSM1k-style fresh test sets), and
group-aware splitting (GroupKFold) and purged/embargoed
cross-validation are standard advice in finance and PHM. What does
not exist is a tool that maps window-level overlap graphs for time
series and attributes the score inflation to each leakage component
via removal replay. Existing practice answers "is there leakage?"
with a total; this answers "which connected region of the split
leaked, and what did each region buy the score?" — and it separates
that from the adjacency premium that purging cannot fix. Novelty is
the mechanism combination, stated plainly.

## 4. The trajectory simulator and windowing

Seeded fleets of 24 units; each unit lives 90–170 timesteps and
fails. Two sensor channels (vibration, temperature) carry a
degradation signal growing quadratically toward end of life, a slow
periodic component, Gaussian noise, and per-unit baselines/offsets.
Windows are sliced at five geometries — length × stride 10×5, 20×5,
20×10, 30×10, and 20×20 — with per-channel summary features (mean,
std, slope, last, min, max) and the remaining useful life at the
window's end as label. The 20×20 geometry is a **control**: windows
tile without overlapping, so shared-timestep leakage is zero by
construction.

## 5. Headline results (measured)

Pooled over 120 seeded runs (24 seeds × 5 geometries, random 75/25
splits):

- **76.1% of test windows leak** on average (± 38.3 pooled across
  all runs; per geometry 89.6–99.9% at every overlapping geometry
  with per-geometry spread at most ± 4.1, 0.0% at the control, as
  constructed). Mean
  leakage mass among leaked windows is 0.604.
- **The overlap graph detects the planted leakage at precision
  1.000 / recall 1.000.** The exact-duplicate-counting baseline —
  the standard quick contamination check — has recall **0.000**: a
  window shifted by one timestep is never an exact duplicate, so the
  hash check is structurally blind to partial overlap.
- **The leaky score is optimistic by 1.01 ± 0.84 RMSE** (leaky
  13.64 vs
  LOUO 14.65; purged 15.04, unit-disjoint split reference 14.59) —
  6.9% of the honest score, pooled. Per geometry the optimism is
  1.94 RMSE at 20×5, 1.27 at 10×5, 0.69 at 30×10, 0.73 at 20×10.
- **Removal replay recovers 86.4% of the MSE optimism**, component
  by component (78.0% at 10×5, 98.1% at 20×5, 83.0% at 20×10, 100.5%
  at 30×10 — slightly over 100% there because kNN neighbour sets
  interact across components; the overshoot is reported, not
  clipped). The pooled share is computed by total squared-error
  gain across runs, not as a simple average of the per-geometry
  shares. Per-window MSE optimism is 27.09 for leaked test windows
  vs 9.26 for unleaked ones.
- **The control proves the attribution is specific:** at 20×20 the
  leaky score is still optimistic by 0.43 RMSE, but the replay
  attributes it to *no* component (recovery 0.0%) — that residue is
  pure same-unit adjacency, visible only against the LOUO score.

Demo (seed 7, 20×5, `python -m window_leakage demo`): 578 windows,
100.0% of test windows leaked (mean mass 0.924), leaky RMSE 9.69 vs
LOUO 11.50 — 1.80 RMSE of optimism, 93.5% of it recovered by the
replay; the single largest component is one unit whose one test
window's neighbours nearly determine its label (test RMSE 35.00 →
48.00 when its leaked train windows are removed).

## 6. Honest limits

- **Synthetic fleets, one simple predictor.** The trajectories are
  simulated with a clean quadratic degradation law, and the
  predictor is kNN — chosen because its predictions are literally
  averages of neighbour labels, so inflation is attributable to the
  leaked neighbours themselves. Absolute optimism values will differ
  for deep models and real C-MAPSS-style data; the instrument
  (graph, components, replay, LOUO ground truth) is the claim, not
  the constants.
- **Detection is exact partly because indices are known.** The
  cartographer audits your own pipeline, where each window's unit
  and timestep range are standard metadata. The near-duplicate
  channel exists for the harder case, but in this testbed it fires
  rarely and the headline precision/recall are carried by the
  overlap channel.
- **Cross-unit leakage is out of scope by construction.** Edges are
  only built within a unit; duplicated trajectories across units
  (a real failure mode in scraped datasets) are not modelled.
- **Purging is not a free fix.** At 20×10 the purged score is
  *worse* than the leaky one by 0.81 RMSE on average:
  purging also removes legitimate neighbours and shrinks the
  training set, which can help or hurt by chance. The LOUO score,
  not the purged score, is the ground truth used for the headline.
- **The adjacency premium is real and unfixable by auditing.** Even
  with zero shared timesteps, a random split scored 0.43 RMSE
  optimistic in the control geometry. If the claim is "generalises
  to new units", only a unit-disjoint split supports it.

## 7. How to run

```bash
pip install pytest streamlit   # runtime itself is stdlib only
python -m window_leakage demo                    # one audited split
python -m window_leakage evaluate --out results  # full evaluation
python scripts/reproduce_results.py              # drift guard
python scripts/build_cases.py                    # rebuild data/
pytest -q                                        # test suite
streamlit run app.py                             # dashboard
```

## 8. Repository layout

- `window_leakage/` — simulator (`generate`), windowing/splitting
  (`windows`), overlap graph (`graph`), predictor (`predictor`),
  exact-duplicate baseline (`baseline`), the audit (`cartographer`),
  evaluation (`evaluate`), reports (`report`), CLI (`cli`).
- `data/` — demo fleet summary and evaluation scenarios (built by
  `scripts/build_cases.py`; trajectories are regenerated from seeds,
  never stored).
- `results/` — stored evaluation: `evaluation.json`, `evaluation.csv`,
  `evaluation_summary.md`, `HEADLINE.md`.
- `notebooks/` — runnable demo and evaluation experiments.
- `examples/` — a minimal audit script and a real captured report.
- `tests/` — the test suite, including a stored-results guard.
- `app.py` — Streamlit dashboard (Leakage map / Components /
  Evaluation).

Author: Amna Raheeba. License: MIT.
