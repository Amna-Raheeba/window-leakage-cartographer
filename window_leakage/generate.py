"""Seeded run-to-failure trajectory simulator.

Each unit lives a random lifetime, then fails. Two sensor channels
(vibration, temperature) carry a degradation signal that grows
quadratically as the unit approaches end of life, plus a slow periodic
component and Gaussian noise. Units differ in baseline level and
offset, so windows from different units are not interchangeable -
which is exactly why leaking windows from the *same* unit across a
split inflates the score so much: the model has effectively seen the
test window's own trajectory.

Everything is driven by random.Random(seed): same seed, same fleet,
on every machine and every Python version.
"""

import math
import random

from .models import Trajectory

MIN_LIFETIME = 90
MAX_LIFETIME = 170


def generate_unit(rng, unit_id):
    lifetime = rng.randint(MIN_LIFETIME, MAX_LIFETIME)
    vib_base = rng.uniform(0.8, 1.4)
    vib_offset = rng.uniform(-0.15, 0.15)
    temp_base = rng.uniform(55.0, 65.0)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    vib = []
    temp = []
    for t in range(lifetime):
        frac = t / float(lifetime)
        deg = frac * frac
        v = (
            vib_base
            + vib_offset
            + 2.8 * deg
            + 0.12 * math.sin(0.35 * t + phase)
            + rng.gauss(0.0, 0.07)
        )
        tp = (
            temp_base
            + 26.0 * deg
            + 0.5 * math.sin(0.11 * t + phase)
            + rng.gauss(0.0, 0.45)
        )
        vib.append(v)
        temp.append(tp)
    return Trajectory(
        unit_id=unit_id, channels=(tuple(vib), tuple(temp)), lifetime=lifetime
    )


def generate_fleet(seed, n_units=24):
    """A fleet of n_units trajectories, fully determined by seed."""
    rng = random.Random(seed)
    return [generate_unit(rng, unit_id) for unit_id in range(n_units)]
