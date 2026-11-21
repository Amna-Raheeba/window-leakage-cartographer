from window_leakage.models import (
    Trajectory,
    Window,
    mean,
    mse,
    rmse,
    segment_of,
    std,
    variance,
)


def test_mean_basic():
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_mean_empty():
    assert mean([]) == 0.0


def test_variance_and_std():
    assert variance([2.0, 2.0, 2.0]) == 0.0
    assert std([1.0, 3.0]) == 1.0


def test_mse_rmse():
    assert mse([3.0, 4.0]) == 12.5
    assert rmse([3.0, 4.0]) == 12.5 ** 0.5
    assert rmse([]) == 0.0


def test_window_end():
    w = Window(window_id=0, unit_id=1, start=10, length=20,
               features=(0.0,), label=5.0)
    assert w.end == 30


def test_segment_of():
    traj = Trajectory(
        unit_id=0,
        channels=(tuple(float(i) for i in range(10)),
                  tuple(float(100 + i) for i in range(10))),
        lifetime=10,
    )
    seg = segment_of(traj, 2, 3)
    assert seg[0] == (2.0, 3.0, 4.0)
    assert seg[1] == (102.0, 103.0, 104.0)
