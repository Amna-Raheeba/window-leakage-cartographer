from window_leakage.generate import generate_fleet
from window_leakage.models import FEATURE_NAMES
from window_leakage.windows import (
    build_windows,
    random_split,
    unit_disjoint_split,
    window_features,
)


def test_window_count_matches_formula():
    fleet = generate_fleet(2, 3)
    length, stride = 10, 5
    windows = build_windows(fleet, length, stride)
    expected = sum(
        (t.lifetime - length) // stride + 1 for t in fleet
    )
    assert len(windows) == expected


def test_window_labels_are_rul_at_end():
    fleet = generate_fleet(2, 3)
    windows = build_windows(fleet, 10, 5)
    by_unit = dict((t.unit_id, t) for t in fleet)
    for w in windows:
        assert w.label == float(by_unit[w.unit_id].lifetime - w.end)
        assert w.label >= 0.0


def test_window_starts_follow_stride():
    fleet = generate_fleet(2, 1)
    windows = build_windows(fleet, 10, 7)
    starts = [w.start for w in windows]
    assert starts[0] == 0
    assert all(b - a == 7 for a, b in zip(starts, starts[1:]))


def test_window_features_shape():
    fleet = generate_fleet(2, 1)
    windows = build_windows(fleet, 10, 5)
    for w in windows:
        assert len(w.features) == len(FEATURE_NAMES) == 12


def test_window_features_constant_segment():
    seg = ((1.0,) * 6, (2.0,) * 6)
    feats = window_features(seg)
    assert feats[0] == 1.0 and feats[1] == 0.0 and feats[2] == 0.0
    assert feats[3] == 1.0 and feats[4] == 1.0 and feats[5] == 1.0


def test_random_split_partitions():
    fleet = generate_fleet(2, 6)
    windows = build_windows(fleet, 10, 5)
    train, test = random_split(windows, 0.25, seed=3)
    train_ids = set(w.window_id for w in train)
    test_ids = set(w.window_id for w in test)
    assert not (train_ids & test_ids)
    assert train_ids | test_ids == set(w.window_id for w in windows)
    assert abs(len(test) - 0.25 * len(windows)) <= 1.0


def test_random_split_deterministic():
    fleet = generate_fleet(2, 6)
    windows = build_windows(fleet, 10, 5)
    a_train, a_test = random_split(windows, 0.25, seed=3)
    b_train, b_test = random_split(windows, 0.25, seed=3)
    assert [w.window_id for w in a_test] == [w.window_id for w in b_test]
    assert [w.window_id for w in a_train] == [w.window_id for w in b_train]


def test_unit_disjoint_split_no_shared_units():
    fleet = generate_fleet(2, 8)
    windows = build_windows(fleet, 10, 5)
    train, test = unit_disjoint_split(windows, 0.25, seed=3)
    train_units = set(w.unit_id for w in train)
    test_units = set(w.unit_id for w in test)
    assert not (train_units & test_units)
    assert train_units | test_units == set(range(8))
    assert test  # at least one held-out unit's windows
