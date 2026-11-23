# C-MAPSS FD001 real-data audit — summary

## Data

- Source: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip (NASA Prognostics Data Repository mirror; CMAPSSData.zip inside)
- File: data/train_FD001.txt, md5 259f340bac32ce6fa8894815600fa757, sha256 963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8
- Verified as the genuine FD001 training set: 20631 rows, 26 columns, 100 engine units (1-100), no NaNs. The sha256 also matches the independently published fingerprint for this file.
- Windows: per engine, length x stride geometries below; features are the per-sensor mean and standard deviation over the window (21 sensors, 42 features); label is the true RUL at the window's last cycle, capped at 125.
- Splits: random 75/25 window-level splits, 10 split seeds (0-9). Scores: leaky / purged / leave-own-unit-out (LOUO) / per-component removal replay, computed with the package's graph, components and leakage mass; regressors are kNN (k=3, package semantics) and ridge (lambda=1.0).

## Geometry: window length 30, stride 5 (3586 windows, 16430 graph edges, 0 near-duplicate edges)

Equivalence check (seed 0): the numpy kNN used here reproduces the package's own audit_split on the same split with max abs difference 0.00e+00 across leaky / purged / LOUO RMSE, recovery share and leaked fraction.

| Regressor | Leaked fraction | Leaky RMSE | Purged RMSE | LOUO RMSE | Optimism (LOUO - leaky) | Recovery share (mean of seeds) | Recovery share (pooled by SE gain) | Straddling components |
|---|---|---|---|---|---|---|---|---|
| kNN (k=3) | 1.000 +/- 0.000 (pooled 1.000) | 11.49 +/- 0.61 | 26.58 +/- 1.54 | 22.81 +/- 0.54 | 11.32 +/- 0.42 | 1.000 +/- 0.003 | 1.000 | 100.0 +/- 0.0 |
| Ridge (lambda=1.0) | 1.000 +/- 0.000 (pooled 1.000) | 20.67 +/- 0.41 | 24.42 +/- 1.13 | 21.41 +/- 0.41 | 0.75 +/- 0.02 | 0.995 +/- 0.006 | 0.995 | 100.0 +/- 0.0 |

## Geometry: window length 20, stride 5 (3786 windows, 10758 graph edges, 0 near-duplicate edges)

| Regressor | Leaked fraction | Leaky RMSE | Purged RMSE | LOUO RMSE | Optimism (LOUO - leaky) | Recovery share (mean of seeds) | Recovery share (pooled by SE gain) | Straddling components |
|---|---|---|---|---|---|---|---|---|
| kNN (k=3) | 0.999 +/- 0.001 (pooled 0.999) | 16.07 +/- 0.36 | 25.07 +/- 0.63 | 23.86 +/- 0.49 | 7.79 +/- 0.32 | 1.002 +/- 0.006 | 1.002 | 100.0 +/- 0.0 |
| Ridge (lambda=1.0) | 0.999 +/- 0.001 (pooled 0.999) | 21.22 +/- 0.27 | 22.66 +/- 0.60 | 21.78 +/- 0.26 | 0.56 +/- 0.01 | 0.963 +/- 0.014 | 0.963 | 100.0 +/- 0.0 |

## What the real-data run shows (honest read)

On genuine C-MAPSS FD001 the leakage is total and, for a local model, the optimism is far larger than on the synthetic fleets, not smaller: at window length 30 / stride 5, 100.0% of test windows share raw cycles with training windows (mean leakage mass 0.985), and the random-split kNN score of 11.49 +/- 0.61 RMSE is optimistic by 11.32 +/- 0.42 RMSE against the leave-own-unit-out score of 22.81 +/- 0.54 for the identical test windows; at 20/5 the kNN optimism is 7.79 +/- 0.32 RMSE. The effect is strongly model-dependent: ridge regression on the same splits is optimistic by only 0.75 +/- 0.02 RMSE (30/5) and 0.56 +/- 0.01 RMSE (20/5), because a global linear model cannot exploit near-copy neighbours the way kNN does, so leakage optimism is a property of the split and the model together, not of the split alone. Two testbed behaviours change character on real data: purging, only mildly biased in the synthetic evaluation, collapses here (purged kNN RMSE 26.58 +/- 1.54 at 30/5, worse than the honest LOUO score), because at 100% leakage the purge deletes nearly all of the training data; and the removal replay recovers essentially all of the measured optimism (recovery share pooled by squared-error gain: 1.000 for kNN and 0.995 for ridge at 30/5, 1.002 for kNN and 0.963 for ridge at 20/5), leaving no measurable same-unit adjacency premium, which is expected when there are effectively no unleaked test windows to carry one. The near-duplicate channel never fired on the real data (0 edges); every one of the graph's edges is a true timestep overlap.
