"""Windowing and splitting.

build_windows slices every trajectory into fixed-length sliding
windows. Each window's features are per-channel summary statistics
(mean, std, least-squares slope, last, min, max) and its label is the
remaining useful life at the window's end.

Two splitters:
- random_split: the standard, leaky practice - shuffle all windows
  and hold out a fraction. Overlapping windows from one trajectory
  land on both sides.
- unit_disjoint_split: the honest reference - whole units are held
  out, so no test window shares a trajectory with any train window.
"""

import random

from .models import Window, mean, std


def _slope(values):
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = mean(values)
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0.0:
        return 0.0
    return num / den


def window_features(segment):
    """Summary features for one window's raw channel segments."""
    feats = []
    for channel in segment:
        feats.extend(
            [
                mean(channel),
                std(channel),
                _slope(channel),
                channel[-1],
                min(channel),
                max(channel),
            ]
        )
    return tuple(feats)


def build_windows(fleet, length, stride):
    """All sliding windows over the fleet, in deterministic order."""
    windows = []
    next_id = 0
    for traj in fleet:
        start = 0
        while start + length <= traj.lifetime:
            segment = tuple(
                tuple(channel[start:start + length]) for channel in traj.channels
            )
            windows.append(
                Window(
                    window_id=next_id,
                    unit_id=traj.unit_id,
                    start=start,
                    length=length,
                    features=window_features(segment),
                    label=float(traj.lifetime - (start + length)),
                )
            )
            next_id += 1
            start += stride
    return windows


def random_split(windows, test_fraction=0.25, seed=0):
    """Shuffle windows and hold out a fraction. Returns (train, test)."""
    rng = random.Random(seed)
    ids = [w.window_id for w in windows]
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_fraction))
    test_ids = set(ids[:n_test])
    train = [w for w in windows if w.window_id not in test_ids]
    test = [w for w in windows if w.window_id in test_ids]
    return train, test


def unit_disjoint_split(windows, test_fraction=0.25, seed=0):
    """Hold out whole units. Returns (train, test)."""
    rng = random.Random(seed)
    unit_ids = sorted(set(w.unit_id for w in windows))
    rng.shuffle(unit_ids)
    n_test_units = max(1, int(round(len(unit_ids) * test_fraction)))
    test_units = set(unit_ids[:n_test_units])
    train = [w for w in windows if w.unit_id not in test_units]
    test = [w for w in windows if w.unit_id in test_units]
    return train, test
