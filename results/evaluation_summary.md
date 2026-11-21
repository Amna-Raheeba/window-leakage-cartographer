# Evaluation summary

Pooled over 20 runs (4 seeds x 5 window geometries, 24 units per fleet, random 75/25 splits, kNN k=3 RUL regression):

- Test windows leaked by the random split: 76.7% on average.
- Overlap-graph detection: precision 1.000 / recall 1.000; the exact-duplicate baseline's recall is 0.000.
- Score optimism: leaky RMSE 14.01 vs leave-own-unit-out 14.87 (purged 14.88, unit-disjoint split reference 14.72) - 0.85 RMSE, 5.7% of the honest score.
- Per-component removal replay recovers 78.0% of the MSE optimism; the remainder is the same-unit adjacency premium.
- Mean per-window MSE optimism: 22.70 for leaked test windows vs 4.60 for unleaked ones (the adjacency premium).

Per geometry (window length x stride):
- 10x5: leaked 94.4%, optimism 1.08 RMSE, replay recovery 60.0%.
- 20x5: leaked 99.8%, optimism 1.68 RMSE, replay recovery 96.3%.
- 20x10: leaked 90.0%, optimism 0.35 RMSE, replay recovery 57.3%.
- 30x10: leaked 99.2%, optimism 0.71 RMSE, replay recovery 105.8%.
- 20x20: leaked 0.0%, optimism 0.45 RMSE, replay recovery 0.0%.
