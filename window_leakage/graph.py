"""The window-overlap graph.

Nodes are windows. Two windows from the same unit are joined by an
edge when either:

- overlap channel: their raw timestep ranges [start, end) share at
  least one timestep. Edge weight = shared timesteps / window length.
  This is the leakage a random split creates mechanically.
- near-duplicate channel: their ranges do not overlap, but their raw
  values match within a tolerance band at almost every position
  (fraction >= similarity threshold). This catches leakage that
  survives re-windowing - repeated or near-repeated segments whose
  indices no longer overlap. Edge weight = the matching fraction.

Cross-unit pairs are never joined: with continuous sensor noise, two
different units do not share raw timesteps, and near-identical raw
segments across units do not occur in this testbed (stated as a limit
in the README).

Connected components of this graph that contain both train and test
windows are the leakage components: the connected regions across
which the split leaked. Leakage mass of a test window combines its
edge weights to train windows as 1 - prod(1 - w).
"""

from .models import segment_of, std

DEFAULT_SIM_THRESHOLD = 0.85
DEFAULT_TOL_FRACTION = 0.12


def shared_steps(a, b):
    """Raw timesteps shared by windows a and b (0 across units)."""
    if a.unit_id != b.unit_id:
        return 0
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    return max(0, hi - lo)


def segment_similarity(seg_a, seg_b, tolerances):
    """Fraction of positions where every channel is within tolerance."""
    length = len(seg_a[0])
    if length == 0:
        return 0.0
    matched = 0
    for i in range(length):
        ok = True
        for c in range(len(seg_a)):
            if abs(seg_a[c][i] - seg_b[c][i]) > tolerances[c]:
                ok = False
                break
        if ok:
            matched += 1
    return matched / float(length)


def channel_tolerances(fleet, tol_fraction=DEFAULT_TOL_FRACTION):
    """Per-channel tolerance: a fraction of the channel's pooled std."""
    tols = []
    n_channels = len(fleet[0].channels) if fleet else 0
    for c in range(n_channels):
        pooled = []
        for traj in fleet:
            pooled.extend(traj.channels[c])
        tols.append(max(1e-9, tol_fraction * std(pooled)))
    return tuple(tols)


def build_edges(windows, fleet, sim_threshold=DEFAULT_SIM_THRESHOLD,
                tol_fraction=DEFAULT_TOL_FRACTION):
    """All leakage edges among windows of the same unit.

    Returns a list of dicts: {a, b, weight, overlap_steps, similarity,
    source} where source is "overlap", "near_duplicate", or "both"
    (similarity is only computed when the ranges do not overlap, so
    "both" does not occur in practice; the field is kept explicit).
    """
    by_unit = {}
    for w in windows:
        by_unit.setdefault(w.unit_id, []).append(w)
    tolerances = channel_tolerances(fleet, tol_fraction)
    traj_by_id = dict((t.unit_id, t) for t in fleet)
    edges = []
    for unit_id in sorted(by_unit):
        group = by_unit[unit_id]
        traj = traj_by_id[unit_id]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = group[i]
                b = group[j]
                ov = shared_steps(a, b)
                if ov > 0:
                    edges.append(
                        {
                            "a": a.window_id,
                            "b": b.window_id,
                            "weight": ov / float(a.length),
                            "overlap_steps": ov,
                            "similarity": None,
                            "source": "overlap",
                        }
                    )
                else:
                    seg_a = segment_of(traj, a.start, a.length)
                    seg_b = segment_of(traj, b.start, b.length)
                    sim = segment_similarity(seg_a, seg_b, tolerances)
                    if sim >= sim_threshold:
                        edges.append(
                            {
                                "a": a.window_id,
                                "b": b.window_id,
                                "weight": sim,
                                "overlap_steps": 0,
                                "similarity": sim,
                                "source": "near_duplicate",
                            }
                        )
    return edges


class UnionFind:
    def __init__(self, items):
        self.parent = dict((x, x) for x in items)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def connected_components(window_ids, edges):
    """Connected components as a list of sorted window-id lists."""
    uf = UnionFind(window_ids)
    for e in edges:
        uf.union(e["a"], e["b"])
    groups = {}
    for wid in window_ids:
        groups.setdefault(uf.find(wid), []).append(wid)
    comps = [sorted(members) for members in groups.values()]
    comps.sort(key=lambda members: members[0])
    return comps


def leakage_mass(window_id, edges_by_window, train_ids):
    """Combined leakage of one test window from its train neighbours.

    1 - prod(1 - w) over edges to train windows: one full-overlap
    neighbour gives mass 1.0; several partial neighbours accumulate.
    """
    remaining = 1.0
    for e in edges_by_window.get(window_id, []):
        other = e["b"] if e["a"] == window_id else e["a"]
        if other in train_ids:
            remaining *= 1.0 - e["weight"]
    return 1.0 - remaining


def index_edges_by_window(edges):
    by_window = {}
    for e in edges:
        by_window.setdefault(e["a"], []).append(e)
        by_window.setdefault(e["b"], []).append(e)
    return by_window
