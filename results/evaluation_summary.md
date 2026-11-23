# Evaluation summary

Pooled over 120 runs (24 seeds x 5 window geometries, 24 units per fleet, random 75/25 splits, kNN k=3 RUL regression):

- Test windows leaked by the random split: 76.1% on average (+/- 38.3 across runs).
- Overlap-graph detection: precision 1.000 / recall 1.000; the exact-duplicate baseline's recall is 0.000.
- Score optimism: leaky RMSE 13.64 (+/- 2.42) vs leave-own-unit-out 14.65 (+/- 2.52) (purged 15.04 (+/- 2.29), unit-disjoint split reference 14.59) - 1.01 (+/- 0.84) RMSE, 6.9% of the honest score.
- Per-component removal replay recovers 86.4% of the MSE optimism; the remainder is the same-unit adjacency premium. The recovery share is pooled by total squared-error gain across runs (total replay gain divided by total LOUO squared-error gain), not a simple average of the per-geometry shares.
- Mean per-window MSE optimism: 27.09 for leaked test windows vs 9.26 for unleaked ones (the adjacency premium).

Per geometry (window length x stride):
- 10x5: leaked 92.6% (+/- 1.9), optimism 1.27 (+/- 0.49) RMSE, replay recovery 78.0%.
- 20x5: leaked 99.9% (+/- 0.2), optimism 1.94 (+/- 0.78) RMSE, replay recovery 98.1%.
- 20x10: leaked 89.6% (+/- 4.1), optimism 0.73 (+/- 0.68) RMSE, replay recovery 83.0%.
- 30x10: leaked 98.4% (+/- 1.8), optimism 0.69 (+/- 0.58) RMSE, replay recovery 100.5%.
- 20x20: leaked 0.0% (+/- 0.0), optimism 0.43 (+/- 0.67) RMSE, replay recovery 0.0%.
