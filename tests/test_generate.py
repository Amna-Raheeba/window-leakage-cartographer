from window_leakage.generate import MAX_LIFETIME, MIN_LIFETIME, generate_fleet
from window_leakage.models import mean


def test_fleet_size_and_ids():
    fleet = generate_fleet(3, 6)
    assert len(fleet) == 6
    assert [t.unit_id for t in fleet] == [0, 1, 2, 3, 4, 5]


def test_deterministic_same_seed():
    a = generate_fleet(9, 4)
    b = generate_fleet(9, 4)
    for ta, tb in zip(a, b):
        assert ta.lifetime == tb.lifetime
        assert ta.channels == tb.channels


def test_different_seeds_differ():
    a = generate_fleet(9, 4)
    b = generate_fleet(10, 4)
    assert any(ta.channels != tb.channels for ta, tb in zip(a, b))


def test_lifetime_bounds_and_channel_lengths():
    fleet = generate_fleet(4, 10)
    for t in fleet:
        assert MIN_LIFETIME <= t.lifetime <= MAX_LIFETIME
        assert len(t.channels) == 2
        for channel in t.channels:
            assert len(channel) == t.lifetime


def test_degradation_trend_present():
    fleet = generate_fleet(6, 12)
    early = []
    late = []
    for t in fleet:
        vib = t.channels[0]
        early.extend(vib[:10])
        late.extend(vib[-10:])
    assert mean(late) > mean(early) + 1.0
