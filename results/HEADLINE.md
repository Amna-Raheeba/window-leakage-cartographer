# Headline results

Random 75/25 splits of sliding windows leak 76.7% of test windows on average (pooled over 20 seeded runs, 5 window geometries including a non-overlapping control). The overlap graph detects them at precision 1.000 / recall 1.000; exact-duplicate counting recalls 0.000. The leaky kNN RUL score is optimistic by 0.85 RMSE (leaky 14.01 vs leave-own-unit-out 14.87; purged 14.88, unit-disjoint split reference 14.72), and per-component removal replay recovers 78.0% of the MSE optimism, component by component - the rest is the same-unit adjacency premium.
