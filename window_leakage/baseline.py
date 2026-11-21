"""The baseline: exact-duplicate counting.

The standard quick check for train/test contamination is to hash
each example and count exact matches. For sliding windows over
continuous sensor data this check is structurally blind: two windows
shifted by a single timestep share almost all of their raw data yet
are not exact duplicates of anything, so the hash never matches.

The baseline here hashes each window's exact raw segment (the tuple
of its channel values) and flags a test window only when a train
window has the identical segment. Its detection rate against the
planted ground truth (shared raw timesteps) is measured in the
evaluation - it under-reports by design, which is the point of the
comparison.
"""

from .models import segment_of


def exact_duplicate_flags(train_windows, test_windows, fleet):
    """Set of test window ids whose raw segment exactly equals a
    train window's raw segment."""
    traj_by_id = dict((t.unit_id, t) for t in fleet)
    train_hashes = set()
    for w in train_windows:
        seg = segment_of(traj_by_id[w.unit_id], w.start, w.length)
        train_hashes.add(seg)
    flagged = set()
    for w in test_windows:
        seg = segment_of(traj_by_id[w.unit_id], w.start, w.length)
        if seg in train_hashes:
            flagged.add(w.window_id)
    return flagged
