from window_leakage.graph import (
    build_edges,
    connected_components,
    index_edges_by_window,
    leakage_mass,
    segment_similarity,
    shared_steps,
)
from window_leakage.models import Trajectory, Window


def _w(wid, unit, start, length=10):
    return Window(window_id=wid, unit_id=unit, start=start, length=length,
                  features=(0.0,), label=0.0)


def _traj(unit=0, n=60):
    vib = tuple(0.1 * i for i in range(n))
    temp = tuple(50.0 + 0.05 * i for i in range(n))
    return Trajectory(unit_id=unit, channels=(vib, temp), lifetime=n)


def test_shared_steps_overlap():
    assert shared_steps(_w(0, 0, 0), _w(1, 0, 5)) == 5


def test_shared_steps_disjoint():
    assert shared_steps(_w(0, 0, 0), _w(1, 0, 10)) == 0


def test_shared_steps_cross_unit():
    assert shared_steps(_w(0, 0, 0), _w(1, 1, 0)) == 0


def test_shared_steps_contained():
    assert shared_steps(_w(0, 0, 0, 30), _w(1, 0, 10, 10)) == 10


def test_segment_similarity_identical():
    seg = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    assert segment_similarity(seg, seg, (0.1, 0.1)) == 1.0


def test_segment_similarity_far():
    a = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    b = ((10.0, 10.0, 10.0), (10.0, 10.0, 10.0))
    assert segment_similarity(a, b, (0.1, 0.1)) == 0.0


def test_segment_similarity_partial():
    a = ((0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
    b = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert segment_similarity(a, b, (0.1, 0.1)) == 2.0 / 3.0


def test_build_edges_overlap_weight():
    fleet = [_traj()]
    windows = [_w(0, 0, 0), _w(1, 0, 5), _w(2, 0, 40)]
    edges = build_edges(windows, fleet)
    pair = dict(
        ((e["a"], e["b"]), e) for e in edges
    )
    assert (0, 1) in pair
    assert pair[(0, 1)]["source"] == "overlap"
    assert pair[(0, 1)]["overlap_steps"] == 5
    assert pair[(0, 1)]["weight"] == 0.5


def test_build_edges_no_cross_unit():
    fleet = [_traj(0), _traj(1)]
    windows = [_w(0, 0, 0), _w(1, 1, 0)]
    edges = build_edges(windows, fleet)
    assert edges == []


def test_connected_components_chain():
    edges = [
        {"a": 0, "b": 1, "weight": 0.5},
        {"a": 1, "b": 2, "weight": 0.5},
        {"a": 3, "b": 4, "weight": 0.5},
    ]
    comps = connected_components([0, 1, 2, 3, 4, 5], edges)
    assert [0, 1, 2] in comps
    assert [3, 4] in comps
    assert [5] in comps


def test_leakage_mass_full_overlap():
    edges = [{"a": 0, "b": 1, "weight": 1.0}]
    by_window = index_edges_by_window(edges)
    assert leakage_mass(0, by_window, {1}) == 1.0


def test_leakage_mass_no_train_neighbour():
    edges = [{"a": 0, "b": 1, "weight": 1.0}]
    by_window = index_edges_by_window(edges)
    assert leakage_mass(0, by_window, {2}) == 0.0


def test_leakage_mass_accumulates():
    edges = [
        {"a": 0, "b": 1, "weight": 0.5},
        {"a": 0, "b": 2, "weight": 0.5},
    ]
    by_window = index_edges_by_window(edges)
    assert leakage_mass(0, by_window, {1, 2}) == 0.75
