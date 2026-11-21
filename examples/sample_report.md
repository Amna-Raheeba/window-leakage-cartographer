# Sample report

Real captured output of `python -m window_leakage demo` (equivalently
`python examples/run_audit.py`): one random 75/25 split of the demo
fleet (seed 7, windows 20 long, stride 5), audited end to end.

```text
Window leakage audit - seed 7, windows 20 long, stride 5
578 windows (434 train / 144 test), 1590 graph edges, 24 components, 24 straddling the split.

Leaked test windows: 100.0% of the test set (mean leakage mass 0.924 among them).
Graph detection vs ground truth: precision 1.000, recall 1.000.
Exact-duplicate baseline caught 0 of the leaked test windows.

kNN RUL score (RMSE, lower is better): leaky split 9.69, purged 14.12, leave-own-unit-out 11.50, unit-disjoint split reference 15.83.
Total optimism vs leave-own-unit-out: 1.80 RMSE. Per-component removal replay recovers 93.5% of the MSE optimism; the rest is the same-unit adjacency premium.

Largest leakage components by attributed MSE inflation:
  units 8: 15 train / 1 test windows, mean mass 0.99, test RMSE 35.00 -> 48.00 without its leaked train windows
  units 1: 19 train / 10 test windows, mean mass 0.93, test RMSE 11.03 -> 16.60 without its leaked train windows
  units 4: 18 train / 3 test windows, mean mass 0.99, test RMSE 11.37 -> 15.49 without its leaked train windows
  units 19: 12 train / 6 test windows, mean mass 0.93, test RMSE 9.94 -> 13.45 without its leaked train windows
  units 20: 23 train / 6 test windows, mean mass 0.93, test RMSE 9.86 -> 13.36 without its leaked train windows
```

Reading it: at this geometry every test window shares raw timesteps
with a train window (mean leakage mass 0.924), the exact-duplicate
baseline sees none of it, and the benchmark's 9.69 RMSE is optimistic
by 1.80 RMSE against the leave-own-unit-out score for the identical
test windows. One component (unit 8, a single test window whose
train neighbours nearly determine its label) accounts for the
largest single share of the inflation.
