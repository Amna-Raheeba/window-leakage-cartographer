from window_leakage.baseline import exact_duplicate_flags
from window_leakage.generate import generate_fleet
from window_leakage.models import Window
from window_leakage.windows import build_windows, random_split


def test_baseline_flags_literal_duplicate():
    fleet = generate_fleet(3, 2)
    windows = build_windows(fleet, 10, 5)
    first = windows[0]
    # A second window object covering the exact same raw segment.
    clone = Window(
        window_id=999,
        unit_id=first.unit_id,
        start=first.start,
        length=first.length,
        features=first.features,
        label=first.label,
    )
    flagged = exact_duplicate_flags([first], [clone], fleet)
    assert flagged == {999}


def test_baseline_flags_nothing_on_random_split():
    fleet = generate_fleet(3, 6)
    windows = build_windows(fleet, 10, 5)
    train, test = random_split(windows, 0.25, seed=1)
    assert exact_duplicate_flags(train, test, fleet) == set()
