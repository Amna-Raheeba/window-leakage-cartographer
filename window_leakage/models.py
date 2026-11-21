"""Core data types for the Window Leakage Cartographer.

A Trajectory is one run-to-failure sensor record from one unit
(machine). A Window is one fixed-length slice of a trajectory, with
summary features and a remaining-useful-life (RUL) label taken at the
window's end. The cartographer audits how a train/test split of these
windows leaks: windows are nodes, shared raw timesteps are edges.
"""

from dataclasses import dataclass

CHANNEL_NAMES = ("vibration", "temperature")

FEATURE_NAMES = (
    "vibration_mean",
    "vibration_std",
    "vibration_slope",
    "vibration_last",
    "vibration_min",
    "vibration_max",
    "temperature_mean",
    "temperature_std",
    "temperature_slope",
    "temperature_last",
    "temperature_min",
    "temperature_max",
)


@dataclass
class Trajectory:
    """One unit's run-to-failure record.

    channels: tuple of per-channel series, each a tuple of floats of
    length == lifetime. channels[c][t] is channel c at timestep t.
    """

    unit_id: int
    channels: tuple
    lifetime: int


@dataclass
class Window:
    """One sliding window over one trajectory.

    start is the first timestep (inclusive); end is exclusive, so the
    window covers raw timesteps [start, end). label is the remaining
    useful life at the window's end: lifetime - end.
    """

    window_id: int
    unit_id: int
    start: int
    length: int
    features: tuple
    label: float

    @property
    def end(self):
        return self.start + self.length


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def variance(values):
    values = list(values)
    if not values:
        return 0.0
    mu = mean(values)
    return sum((v - mu) ** 2 for v in values) / float(len(values))


def std(values):
    return variance(values) ** 0.5


def mse(errors):
    """Mean squared error from a list of (prediction - truth) errors."""
    errors = list(errors)
    if not errors:
        return 0.0
    return sum(e * e for e in errors) / float(len(errors))


def rmse(errors):
    return mse(errors) ** 0.5


def segment_of(trajectory, start, length):
    """Raw channel segments of a trajectory over [start, start+length)."""
    return tuple(
        tuple(channel[start:start + length]) for channel in trajectory.channels
    )
