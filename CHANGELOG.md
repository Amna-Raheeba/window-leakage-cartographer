# Changelog

## 0.1.0 - 2026-11-22

- Initial release: seeded run-to-failure trajectory simulator (two
  sensor channels, quadratic degradation) and a sliding-window slicer
  with per-channel summary features and RUL labels.
- Window-overlap graph: edges weighted by shared raw timesteps, plus
  a tolerance-banded near-duplicate channel for non-overlapping
  windows whose raw values nearly coincide; connected components and
  per-test-window leakage mass.
- Removal-replay attribution: a kNN RUL regressor is scored on the
  full training set, on the decontaminated training set, and once per
  leakage component with only that component's train windows removed,
  so the score inflation is priced component by component.
- Baseline comparison: exact-duplicate counting, which structurally
  misses partial overlaps.
- Seeded evaluation over 4 window geometries x 4 seeds with a
  unit-disjoint reference split; Streamlit dashboard (3 tabs), CLI
  demo/evaluate commands, and a reproduction script with a
  headline-drift guard.
